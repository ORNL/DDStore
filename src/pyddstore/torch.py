"""PyTorch integration for DDStore.

- ``DistDataset``: a map-style ``torch.utils.data.Dataset`` backed by DDStore.
  Each rank loads its share of any map-style source dataset into the store;
  every rank can then read any sample. Samples keep the source's structure
  (a tensor/array/number, a tuple or list of them, or a dict of them) and
  each field's shape and dtype. ``__getitems__`` reads a whole batch with one
  ``PyDDStore.get_batch()`` per field, which PyTorch's ``DataLoader`` uses
  automatically.
- ``DistDatasetReader``: the same, as a ``method=2`` extra member that joins a
  dataset published by a ``DistDataset`` core group through a shared
  handshake directory.
- ``ThreadDataLoader``: a ``DataLoader`` whose workers are threads instead of
  forked processes (safe with MPI and GPU-resident buffers).

Fields must have the same shape and dtype in every sample (fixed-shape).
Supported dtypes: bool, uint8, int32, int64, float32, float64.

Import torch before mpi4py/MPI starts; ``from pyddstore.torch import ...``
does that by itself.
"""

import json
import logging
import multiprocessing as mp
import os
import queue
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

__all__ = ["DistDataset", "DistDatasetReader", "ThreadDataLoader"]

_SUPPORTED = {
    np.dtype(np.bool_),
    np.dtype(np.uint8),
    np.dtype(np.int32),
    np.dtype(np.int64),
    np.dtype(np.float32),
    np.dtype(np.float64),
}
_TORCH_DTYPE = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _nsplit(n, parts):
    """Contiguous index ranges [lo, hi) splitting range(n) into `parts`."""
    k, m = divmod(n, parts)
    return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(parts)]


def _handshake_dir(handshake_dir):
    return handshake_dir or os.environ.get("DDSTORE_HANDSHAKE_DIR") or "./ddstore_hs"


def _meta_path(handshake_dir, name):
    # same sanitization as the C library's record files
    safe = name.replace("/", "_").replace(".", "_")
    return os.path.join(handshake_dir, f"{safe}.meta.json")


# ---------------------------------------------------------------------------
# sample structure <-> flat fields
# ---------------------------------------------------------------------------


def _record_descr(dtype):
    """JSON-safe description of a structured dtype (see _record_dtype)."""
    return json.loads(json.dumps(np.lib.format.dtype_to_descr(dtype)))


def _record_dtype(descr):
    """Structured dtype from _record_descr's output (JSON turns tuples into
    lists; numpy needs them back as tuples)."""

    def fix(d):
        if isinstance(d, str):
            return d
        out = []
        for item in d:
            name = tuple(item[0]) if isinstance(item[0], list) else item[0]
            entry = (name, fix(item[1]))
            out.append(entry + (tuple(item[2]),) if len(item) > 2 else entry)
        return out

    return np.lib.format.descr_to_dtype(fix(descr))


def _field_spec(value, where):
    """(kind, numpy dtype, shape) of one leaf value."""
    # numpy structured records: stored as raw bytes (uint8), rebuilt on read
    if isinstance(value, (np.ndarray, np.void)) and value.dtype.names is not None:
        if isinstance(value, np.void):
            kind, shape = "record", ()
        else:
            kind = "recarray" if isinstance(value, np.recarray) else "structarray"
            shape = value.shape
        return {
            "kind": kind,
            "dtype": "uint8",
            "shape": list(shape),
            "record": _record_descr(value.dtype),
        }
    if isinstance(value, torch.Tensor):
        kind, dtype, shape = (
            "torch",
            np.dtype(str(value.dtype).replace("torch.", "")),
            tuple(value.shape),
        )
    elif isinstance(value, np.ndarray):
        kind, dtype, shape = "numpy", value.dtype, value.shape
    elif isinstance(value, np.generic):
        kind, dtype, shape = "npscalar", value.dtype, ()
    elif isinstance(value, bool):
        kind, dtype, shape = "py", np.dtype(np.bool_), ()
    elif isinstance(value, int):
        kind, dtype, shape = "py", np.dtype(np.int64), ()
    elif isinstance(value, float):
        kind, dtype, shape = "py", np.dtype(np.float64), ()
    else:
        raise TypeError(
            f"{where}: unsupported value of type {type(value).__name__}; "
            "use tensors, numpy arrays or Python/numpy numbers"
        )
    if dtype not in _SUPPORTED:
        raise TypeError(
            f"{where}: dtype {dtype} is not supported "
            f"(supported: {', '.join(sorted(str(d) for d in _SUPPORTED))})"
        )
    return {"kind": kind, "dtype": dtype.name, "shape": list(shape)}


def _flatten(sample):
    """(structure, keys, leaf values) of one sample."""
    if isinstance(sample, dict):
        keys = [str(k) for k in sample.keys()]
        return "dict", keys, list(sample.values())
    if isinstance(sample, (tuple, list)):
        structure = "tuple" if isinstance(sample, tuple) else "list"
        return structure, [str(i) for i in range(len(sample))], list(sample)
    return "single", ["0"], [sample]


def _schema_of(sample, where):
    structure, keys, values = _flatten(sample)
    for v in values:
        if isinstance(v, (dict, list, tuple)):
            raise TypeError(f"{where}: nested containers are not supported")
    fields = [_field_spec(v, f"{where} field {k!r}") for k, v in zip(keys, values)]
    return {"structure": structure, "keys": keys, "fields": fields}


def _to_numpy_row(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    if isinstance(value, (np.ndarray, np.void)) and value.dtype.names is not None:
        return np.frombuffer(np.array(value, dtype=value.dtype).tobytes(), np.uint8)
    return np.asarray(value).reshape(-1)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


class _StoreDataset(Dataset):
    """Reads samples described by `self.schema` from `self.ddstore`."""

    def _setup_fields(self, schema, name, device):
        self.schema = schema
        self.name = name
        self.device = device
        self._var = [f"{name}/{k}" for k in schema["keys"]]
        self._record = [
            _record_dtype(f["record"]) if "record" in f else None
            for f in schema["fields"]
        ]
        self._size = [
            int(np.prod(f["shape"], dtype=np.int64))
            * (rec.itemsize if rec is not None else 1)
            for f, rec in zip(schema["fields"], self._record)
        ]
        self._batch_get = os.environ.get("DDSTORE_BATCH_GET", "1") != "0"

    # -- shapes as the user sees them (same structure as a sample) --------
    @property
    def shapes(self):
        return self._rebuild([tuple(f["shape"]) for f in self.schema["fields"]])

    @property
    def dtypes(self):
        # plain fields: dtype name; record fields: the structured numpy dtype
        return self._rebuild(
            [
                rec if rec is not None else f["dtype"]
                for f, rec in zip(self.schema["fields"], self._record)
            ]
        )

    def _rebuild(self, values):
        s = self.schema["structure"]
        if s == "single":
            return values[0]
        if s == "dict":
            return dict(zip(self.schema["keys"], values))
        return tuple(values) if s == "tuple" else list(values)

    def _alloc(self, n, j):
        f = self.schema["fields"][j]
        if self.device is not None and f["kind"] == "torch":
            return torch.empty(
                (n, self._size[j]), dtype=_TORCH_DTYPE[f["dtype"]], device=self.device
            )
        return np.empty((n, self._size[j]), dtype=f["dtype"])

    def _value(self, row, j):
        """One sample's field from its flat row (a view, no copy)."""
        f = self.schema["fields"][j]
        shape = tuple(f["shape"])
        if f["kind"] == "torch":
            t = row if isinstance(row, torch.Tensor) else torch.from_numpy(row)
            return t.reshape(shape)
        if f["kind"] == "numpy":
            return row.reshape(shape)
        if f["kind"] == "record":
            return row.view(self._record[j])[0]
        if f["kind"] in ("structarray", "recarray"):
            arr = row.view(self._record[j]).reshape(shape)
            return arr.view(np.recarray) if f["kind"] == "recarray" else arr
        if f["kind"] == "npscalar":
            return row[0]
        return row[0].item()

    def __len__(self):
        return self.total_ns

    def len(self):
        return self.total_ns

    def get(self, idx):
        values = []
        for j, var in enumerate(self._var):
            buf = self._alloc(1, j)
            self.ddstore.get(var, buf, int(idx))
            values.append(self._value(buf[0], j))
        return self._rebuild(values)

    def __getitem__(self, idx):
        return self.get(idx)

    def __getitems__(self, indices):
        """A whole batch: one get_batch() per field (DDSTORE_BATCH_GET=0:
        one get() per sample). Called by DataLoader and ThreadDataLoader."""
        if not self._batch_get:
            return [self.get(i) for i in indices]
        idx = np.asarray(indices, dtype=np.int64)
        columns = []
        for j, var in enumerate(self._var):
            buf = self._alloc(len(idx), j)
            self.ddstore.get_batch(var, buf, idx)
            columns.append([self._value(buf[i], j) for i in range(len(idx))])
        return [self._rebuild([col[i] for col in columns]) for i in range(len(idx))]


def _rows(values, dtype):
    """Stack per-sample values into contiguous (n, size) rows of `dtype`."""
    return np.ascontiguousarray(
        np.stack([_to_numpy_row(v) for v in values]).astype(dtype, copy=False)
    )


class DistDataset(_StoreDataset):
    """A map-style dataset stored in DDStore across the ranks of `comm`.

    Args:
        source: any map-style dataset (``len(source)``, ``source[i]``). Each
            rank loads only its contiguous share. Every sample must have the
            same structure, and each field the same shape and dtype.
        name: dataset name; field ``k`` is stored as variable ``name/k``.
        comm: MPI communicator (default ``MPI.COMM_WORLD``). All its ranks
            must construct the dataset together.
        ddstore_width: ranks per independent store (default: all of
            ``comm``); each group holds a full copy of the dataset.
        device: put tensor fields of read samples on this device
            (GPUDirect RDMA; needs ``method`` 1/2 and ``DDSTORE_FABRIC=cxi``).
        add_device: keep this rank's share of tensor fields on this device.
        method: DDStore backend (default ``DDSTORE_METHOD`` or 0).
        handshake_dir: ``method=2`` directory (default
            ``DDSTORE_HANDSHAKE_DIR`` or ``./ddstore_hs``).
        chunk_size: load this rank's share ``chunk_size`` samples at a time,
            writing each chunk into the store before reading the next, so
            only one chunk is held in memory besides the store (default:
            read the whole share, then add it). Host storage only (no
            ``add_device`` for tensor fields).

    ``ds[i]`` returns a sample with the source's structure: tensors stay
    tensors (on ``device`` if given), numpy arrays stay arrays, numbers stay
    numbers. ``ds.ddstore`` is the underlying ``PyDDStore``.

    With ``method=0``, batched reads are collective: every rank must read the
    same number of batches (DistributedSampler does that) from one thread.
    """

    def __init__(
        self,
        source,
        name,
        comm=None,
        ddstore_width=None,
        device=None,
        add_device=None,
        method=None,
        handshake_dir=None,
        chunk_size=None,
    ):
        super().__init__()
        from mpi4py import MPI

        from ._core import PyDDStore

        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.add_device = add_device
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        self.method = (
            int(os.environ.get("DDSTORE_METHOD", "0"))
            if method is None
            else int(method)
        )
        if self.method == 2 and self.ddstore_width != self.comm_size:
            raise NotImplementedError(
                "method=2 does not support ddstore_width < comm size (groups would "
                "collide on the same handshake directory)"
            )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        group_rank = self.ddstore_comm.Get_rank()
        group_size = self.ddstore_comm.Get_size()

        # This rank's share of the source. Errors found locally are raised
        # only after comparing with every rank (_raise_on_all), so a bad
        # sample on some ranks raises on all of them instead of leaving the
        # others blocked in a collective.
        self.total_ns = len(source)
        lo, hi = _nsplit(self.total_ns, group_size)[group_rank]
        first, schema, error = None, None, None
        try:
            if hi > lo:
                first = source[lo]
                schema = _schema_of(first, f"{name}[{lo}]")
        except (TypeError, ValueError) as exc:
            error = exc
        schemas = self._raise_on_all(error, schema)
        if any(s is None for s in schemas):
            raise ValueError(
                f"{name}: every rank needs at least one sample (dataset has {self.total_ns})"
            )
        if any(s != schemas[0] for s in schemas):
            raise ValueError(
                f"{name}: samples differ in structure, shape or dtype across ranks"
            )
        self._setup_fields(schemas[0], name, device)
        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError(f"chunk_size must be >= 1 (got {chunk_size})")
            if add_device is not None and any(
                f["kind"] == "torch" for f in self.schema["fields"]
            ):
                raise ValueError(
                    "chunk_size needs host storage: it can't be combined with add_device"
                )

        def check(i, sample):
            other = _schema_of(sample, f"{name}[{i}]")
            if other != self.schema:
                raise ValueError(
                    f"{name}[{i}]: structure, shape or dtype differs from {name}[{lo}] "
                    f"({other} vs {self.schema}); DistDataset needs fixed-shape samples"
                )

        hs = _handshake_dir(handshake_dir)
        if self.method == 2:
            self.ddstore = PyDDStore(self.ddstore_comm, method=2, handshake_dir=hs)
            if group_rank == 0:
                # published before the variables, so a reader that sees a
                # variable's record file always finds the schema too
                path = _meta_path(hs, name)
                tmp = f"{path}.tmp.{os.getpid()}"
                with open(tmp, "w") as fh:
                    json.dump({"total_ns": self.total_ns, **self.schema}, fh)
                os.replace(tmp, path)
        else:
            self.ddstore = PyDDStore(self.ddstore_comm, method=self.method)

        fields = self.schema["fields"]
        if chunk_size is None:
            # Whole share at once: read, check, then add() each field.
            samples, error = [first], None
            try:
                for i in range(lo + 1, hi):
                    samples.append(source[i])
                    check(i, samples[-1])
            except (TypeError, ValueError) as exc:
                error = exc
            self._raise_on_all(error)
            for j, var in enumerate(self._var):
                values = [_flatten(s)[2][j] for s in samples]
                if add_device is not None and fields[j]["kind"] == "torch":
                    rows = (
                        torch.stack([v.reshape(-1) for v in values])
                        .to(add_device)
                        .contiguous()
                    )
                else:
                    rows = _rows(values, fields[j]["dtype"])
                self.ddstore.add(var, rows)
        else:
            # Chunked: allocate every field (init, collective), then copy the
            # share in chunks of chunk_size samples (update, local), so at most
            # one chunk is held in memory besides the store itself.
            for j, var in enumerate(self._var):
                itemsize = np.dtype(fields[j]["dtype"]).itemsize
                self.ddstore.init(var, hi - lo, self._size[j], itemsize)
            error = None
            try:
                for start in range(lo, hi, chunk_size):
                    stop = min(start + chunk_size, hi)
                    chunk = [
                        first if i == lo else source[i] for i in range(start, stop)
                    ]
                    for i, sample in zip(range(start, stop), chunk):
                        if i != lo:
                            check(i, sample)
                    for j, var in enumerate(self._var):
                        values = [_flatten(sm)[2][j] for sm in chunk]
                        self.ddstore.update(
                            var, _rows(values, fields[j]["dtype"]), start - lo
                        )
                    if start == lo:
                        first = None  # held only for the first chunk
            except (TypeError, ValueError) as exc:
                error = exc
            # also makes sure every rank has filled its share before any reads
            self._raise_on_all(error)
        logger.debug(
            "DistDataset %s: rank %d holds [%d, %d) of %d",
            name,
            self.rank,
            lo,
            hi,
            self.total_ns,
        )

    def _raise_on_all(self, error, value=None):
        """Allgather (value, error) over comm; if any rank had an error,
        raise it on every rank. Returns the gathered values."""
        local = None if error is None else (type(error).__name__, str(error))
        gathered = self.comm.allgather((value, local))
        errors = [e for _, e in gathered if e is not None]
        if errors:
            cls = TypeError if errors[0][0] == "TypeError" else ValueError
            raise cls(errors[0][1])
        return [v for v, _ in gathered]


class DistDatasetReader(_StoreDataset):
    """A ``DistDataset`` published by a ``method=2`` core group, joined from a
    separate job (no MPI communicator needed).

    Args:
        name: the core group's dataset name.
        handshake_dir: shared directory (default ``DDSTORE_HANDSHAKE_DIR`` or
            ``./ddstore_hs``).
        n_core: number of core ranks (default ``DDSTORE_N_CORE``).
        device: put tensor fields of read samples on this device.

    Waits up to ``DDSTORE_HANDSHAKE_TIMEOUT_S`` (default 300 s) for the core
    group to publish.
    """

    def __init__(self, name, handshake_dir=None, n_core=None, device=None):
        super().__init__()
        from ._core import PyDDStore

        hs = _handshake_dir(handshake_dir)
        if n_core is None:
            if "DDSTORE_N_CORE" not in os.environ:
                raise ValueError(
                    "DistDatasetReader needs the number of core ranks: pass n_core= "
                    "or set DDSTORE_N_CORE"
                )
            n_core = int(os.environ["DDSTORE_N_CORE"])
        timeout = float(os.environ.get("DDSTORE_HANDSHAKE_TIMEOUT_S", "300"))
        path = _meta_path(hs, name)
        t0 = time.monotonic()
        while not os.path.exists(path):
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"no dataset {name!r} published in {hs} after {timeout:.0f} s"
                )
            time.sleep(0.05)
        with open(path) as fh:
            meta = json.load(fh)
        self.total_ns = meta.pop("total_ns")
        self._setup_fields(meta, name, device)
        self.ddstore = PyDDStore(None, method=2, handshake_dir=hs, n_core=n_core)
        for var in self._var:
            self.ddstore.join(var)


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


class ThreadDataLoader(DataLoader):
    """A ``DataLoader`` that fetches batches in a thread pool instead of forked
    worker processes. Threads share the process's MPI state, CUDA context and
    Python objects, so it is safe with DDStore and GPU-resident buffers, where
    forked workers are not. Takes the same arguments as ``DataLoader``;
    ``num_workers`` is the number of threads (0 means 1).

    Each batch is fetched (via ``dataset.__getitems__`` when present),
    collated and optionally pinned in a worker thread; at most
    ``num_workers * prefetch_factor`` batches are in flight. Random draws
    match ``DataLoader``'s. ``DDSTORE_AFFINITY_WIDTH`` /
    ``DDSTORE_AFFINITY_OFFSET`` pin worker thread *i* to CPUs
    ``[offset + i*width, offset + (i+1)*width)`` of the process's affinity.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        # Persistent across epochs -- recreating the pool in every __iter__()
        # would leak OS threads since the old pool is never shut down.
        self._counter = mp.Value("i", 0)
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_workers or 1,
            initializer=self.worker_init,
            initargs=(self._counter,),
        )

        logger.debug("num_workers: %s", self.num_workers)
        logger.debug("len: %s", len(self._index_sampler))

    @staticmethod
    def worker_init(counter):
        core_width = int(os.environ.get("DDSTORE_AFFINITY_WIDTH", "0"))
        core_offset = int(os.environ.get("DDSTORE_AFFINITY_OFFSET", "0"))
        if core_width <= 0 or not hasattr(os, "sched_getaffinity"):
            return 0

        with counter.get_lock():
            wid = counter.value
            counter.value += 1

        affinity = list(os.sched_getaffinity(0))
        affinity_mask = set(
            affinity[
                core_width * wid + core_offset : core_width * (wid + 1) + core_offset
            ]
        )
        if affinity_mask:
            os.sched_setaffinity(0, affinity_mask)
        logger.debug(
            "Worker: pid=%s hostname=%s ID=%s affinity=%s",
            os.getpid(),
            socket.gethostname(),
            wid,
            os.sched_getaffinity(0),
        )
        return 0

    @staticmethod
    def fetch(
        dataset, ibatch, index, collate_fn=None, pin_memory=False, auto_collation=True
    ):
        # Collate here, in the worker, before pinning: pinning per-sample
        # tensors and collating afterwards would just torch.stack them into
        # a new, unpinned tensor. Use the dataset's whole-batch fetch when it
        # has one, like torch's own map-style fetcher. With batch_size=None
        # (no auto-collation) the sampler's index goes to dataset[index] as
        # is, also as torch does (e.g. samplers that yield whole batches).
        if not auto_collation:
            batch = dataset[index]
        elif getattr(dataset, "__getitems__", None):
            batch = dataset.__getitems__(index)
        else:
            batch = [dataset[i] for i in index]
        if collate_fn is not None:
            batch = collate_fn(batch)
        if pin_memory:
            batch = torch.utils.data._utils.pin_memory.pin_memory(batch)
        return (ibatch, batch)

    def __iter__(self):
        """A new iterator over one epoch. Each has its own sampler position
        and queue (the thread pool is shared), so several iterators over one
        loader don't interfere, as with ``DataLoader``."""
        return _ThreadLoaderIter(self)

    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False)


class _ThreadLoaderIter:
    """One epoch of a ThreadDataLoader; ``__iter__`` returns itself."""

    def __init__(self, loader):
        self.loader = loader
        self._sampler_iter = iter(loader._index_sampler)
        # torch's DataLoader iterator draws a base seed from the global RNG
        # here, every epoch; draw it too, so the training loop's later random
        # draws are the same as with DataLoader
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
        self.fs = queue.Queue()
        self.fs_iter = iter(self.fs.get, None)
        self._num_yielded = 0
        self._next_batch_i = 0
        self._inflight = 0
        self._sampler_exhausted = False
        # Bound how many batches can be in flight (submitted but not yet
        # consumed via __next__) at once, instead of submitting the whole
        # epoch up front -- keeps memory use (GPU tensors included) bounded
        # regardless of dataset size. Mirrors torch's own prefetch_factor
        # (default 2 per worker).
        self._max_inflight = max(
            1, (loader.num_workers or 1) * (loader.prefetch_factor or 2)
        )
        self._refill()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)

    def _refill(self):
        loader = self.loader
        while self._inflight < self._max_inflight:
            try:
                index = next(self._sampler_iter)
            except StopIteration:
                if not self._sampler_exhausted:
                    self._sampler_exhausted = True
                    self.fs.put(None)
                return
            future = loader.executor.submit(
                loader.fetch,
                loader.dataset,
                self._next_batch_i,
                index,
                collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory,
                auto_collation=loader._auto_collation,
            )
            self.fs.put(future)
            self._next_batch_i += 1
            self._inflight += 1

    def __next__(self):
        # Refill *before* popping this call's batch, not after: refilling
        # here only uses capacity freed by the *previous* call's batch,
        # which -- by ordinary for-loop semantics -- the caller's loop body
        # has already fully consumed by the time it asks for the next item
        # (i.e. calls __next__ again). Bounds how far the executor can race
        # ahead of consumption (memory, not correctness -- each batch is read
        # into its own freshly allocated buffers).
        self._refill()
        future = next(self.fs_iter)
        ibatch, data = future.result()
        self._inflight -= 1
        self._num_yielded += 1
        return data

    def close(self):
        """Cancel the batches still queued (an epoch stopped early)."""
        # Without blocking: the end marker (None) is queued only once the
        # sampler is exhausted, so an epoch that stopped early has none, and
        # waiting for it (iter(self.fs.get, None)) would block forever. Only
        # this thread puts into fs, so qsize() is exact here.
        while self.fs.qsize() > 0:
            future = self.fs.get_nowait()
            if future is not None:
                future.cancel()

    def __del__(self):
        if hasattr(self, "fs"):
            self.close()
