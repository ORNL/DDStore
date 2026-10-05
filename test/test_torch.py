"""
pyddstore.torch (DistDataset, DistDatasetReader, ThreadDataLoader) tests —
run with 2+ ranks, e.g.:
  mpirun -n 4 pytest test/test_torch.py -v

Method 0 always; method 1 and the method-2 reader where a CXI device is
present inside a Slurm step (provider: DDSTORE_FABRIC, default cxi).
"""

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from pyddstore.torch import (
    DistDataset,
    DistDatasetReader,
    ThreadDataLoader,
    WindowedDataset,
    row_of,
)  # noqa: E402

HAVE_CXI = bool(glob.glob("/dev/cxi*")) and "SLURM_STEP_ID" in os.environ
FABRIC = os.environ.get("DDSTORE_FABRIC", "cxi")
HAVE_GPU = torch.cuda.is_available()
METHODS = [
    0,
    pytest.param(1, marks=pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")),
]
N = 37  # not a multiple of the rank count


class TupleSource(Dataset):
    """Every field kind, values derived from the index."""

    def __len__(self):
        return N

    def __getitem__(self, i):
        return (
            torch.arange(12, dtype=torch.float32).reshape(3, 4)
            + 100 * i,  # torch tensor
            i,  # Python int
            np.full(2, i / 3, dtype=np.float64),  # numpy array
            np.int32(-i),  # numpy scalar
            i % 2 == 0,  # Python bool
            float(i) * 0.5,  # Python float
            torch.tensor([i, i + 1], dtype=torch.uint8),  # small dtype
        )


class DictSource(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return {
            "x": torch.full((2, 3), float(i)),
            "y": torch.tensor(i, dtype=torch.int64),
        }


class SingleSource(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return np.full((5,), i, dtype=np.int32)


REC = np.dtype(
    [
        ("x_modules", np.float32, (4, 3)),
        ("mask", np.bool_, (4,)),
        ("params", np.int64, (2,)),
    ]
)
# padded (align=True) and nested layout
REC_NESTED = np.dtype(
    [
        ("a", np.uint8),
        ("b", np.float64),
        ("sub", [("c", np.int32, (2,)), ("d", np.bool_)]),
    ],
    align=True,
)


def _record(i, dtype=REC):
    a = np.zeros((), dtype=dtype)
    if dtype is REC:
        a["x_modules"], a["mask"], a["params"] = i, i % 2 == 0, (i, -i)
    else:
        a["a"], a["b"], a["sub"]["c"], a["sub"]["d"] = (
            i % 256,
            i / 7,
            (i, 2 * i),
            i % 3 == 0,
        )
    return a[()]  # np.void


class RecordSource(Dataset):
    """Items are numpy structured records, in the forms projects use."""

    def __init__(self, form):
        self.form = form

    def __len__(self):
        return N

    def __getitem__(self, i):
        if self.form == "void":
            return _record(i)
        if self.form == "array":  # 1-element structured ndarray
            return np.array([_record(i)], dtype=REC)
        if self.form == "recarray":  # np.recarray of shape (2,)
            return np.array([_record(i), _record(i + 1)], dtype=REC).view(np.recarray)
        if self.form == "nested":
            return _record(i, REC_NESTED)
        return {"rec": _record(i), "t": torch.full((3,), float(i))}  # mixed dict


def same(a, b):
    """Equal structure, types and values."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and bool(torch.equal(a.cpu(), b.cpu()))
        )
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a == b
    return a == b


def all_ok(comm, ok):
    return comm.allreduce(int(bool(ok)), op=__import__("mpi4py").MPI.LAND)


def make(comm, monkeypatch, source, method, **kw):
    if method != 0:
        monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    return DistDataset(source, f"t{method}", comm, method=method, **kw)


def finish(comm, ds):
    comm.Barrier()  # every rank done reading before any rank tears down
    ds.ddstore.free()


@pytest.mark.parametrize("chunk_size", [None, 1, 4])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("source_cls", [TupleSource, DictSource, SingleSource])
def test_items_match_source(comm, monkeypatch, method, source_cls, chunk_size):
    src = source_cls()
    ds = make(comm, monkeypatch, src, method, chunk_size=chunk_size)
    rng = np.random.default_rng(comm.Get_rank())
    idx = rng.integers(0, N, size=20)
    ok = len(ds) == N
    ok &= all(same(ds[int(i)], src[int(i)]) for i in idx[:5])  # per-sample get()
    batch = ds.__getitems__(idx)  # one get_batch() per field (collective for method 0)
    ok &= all(same(b, src[int(i)]) for b, i in zip(batch, idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_shapes_and_dtypes(comm, monkeypatch):
    ds = make(comm, monkeypatch, DictSource(), 0)
    ok = ds.shapes == {"x": (2, 3), "y": ()} and ds.dtypes == {
        "x": "float32",
        "y": "int64",
    }
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("loader", ["DataLoader", "ThreadDataLoader"])
def test_loaders_match_plain_source(comm, monkeypatch, method, loader):
    """A whole epoch through the loader equals the same loader over the plain
    source (same order, same collation). Collective for method 0: every rank
    iterates the same number of batches."""
    src = TupleSource()
    ds = make(comm, monkeypatch, src, method)
    cls = DataLoader if loader == "DataLoader" else ThreadDataLoader
    kw = {} if cls is DataLoader else {"num_workers": 1 if method == 0 else 2}
    got = list(cls(ds, batch_size=8, **kw))
    ref = list(DataLoader(src, batch_size=8))
    ok = len(got) == len(ref) and all(same(g, r) for g, r in zip(got, ref))
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_thread_loader_iterators(comm):
    """iter(loader) is a separate iterator, as with DataLoader: iterating it
    again continues the epoch (list(it), islice), two iterators over one
    loader are independent, and an epoch stopped early doesn't leak into the
    next one."""
    import itertools

    src = TupleSource()
    ref = list(DataLoader(src, batch_size=4))
    loader = ThreadDataLoader(src, batch_size=4, num_workers=2)

    def eq(got, want):
        return len(got) == len(want) and all(same(g, r) for g, r in zip(got, want))

    it = iter(loader)
    ok = iter(it) is it and len(it) == len(ref)
    first = next(it)
    two = list(itertools.islice(it, 2))
    rest = list(it)  # continues, does not restart
    ok &= eq([first] + two + rest, ref)
    a, b = iter(loader), iter(loader)
    got_a, got_b = [], []
    for x, y in zip(a, b):  # interleaved
        got_a.append(x)
        got_b.append(y)
    ok &= eq(got_a, ref) and eq(got_b, ref)
    for i, _ in enumerate(loader):  # stop early
        if i == 1:
            break
    ok &= eq(list(loader), ref)
    assert all_ok(comm, ok)


def test_thread_loader_keeps_prefetch_full(comm):
    """While the caller holds a batch, num_workers * prefetch_factor more are
    being fetched (as with DataLoader), not one fewer."""
    import threading
    import time

    class Counting(Dataset):
        def __init__(self):
            self.lock, self.batches = threading.Lock(), 0

        def __len__(self):
            return 64

        def __getitems__(self, idx):
            with self.lock:
                self.batches += 1
            return [torch.tensor(i) for i in idx]

    src = Counting()
    loader = ThreadDataLoader(src, batch_size=4, num_workers=2, prefetch_factor=1)
    it = iter(loader)
    next(it)  # held by the "training step"
    want = 1 + 2 * 1
    deadline = time.time() + 10
    while src.batches < want and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)  # nothing beyond the bound should start
    ok = src.batches == want
    it.close()
    assert all_ok(comm, ok), f"{src.batches} batches fetched, want {want}"


def stacked(src, rows, key):
    """Field `key` of src[r] for r in rows, stacked as read_rows returns it."""
    vals = [src[int(r)] if key == 0 and not isinstance(src[0], (tuple, list, dict))
            else src[int(r)][key] for r in rows]
    if isinstance(vals[0], torch.Tensor):
        return torch.stack(vals)
    return np.stack([np.asarray(v) for v in vals])


def same_values(a, b):
    a = a.cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b = b.cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def shares(value, buf):
    if isinstance(buf, torch.Tensor):
        return value.data_ptr() == buf.data_ptr()
    return np.shares_memory(np.asarray(value), buf)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("source_cls", [TupleSource, DictSource, SingleSource])
def test_read_rows_and_out(comm, monkeypatch, method, source_cls):
    """read_rows (all fields / a subset), alloc() buffers reused across
    reads, __getitems__(out=). Same number of calls on every rank (method 0
    is collective)."""
    src = source_cls()
    ds = make(comm, monkeypatch, src, method)
    keys = [ds._key(j) for j in range(len(ds._var))]
    rng = np.random.default_rng(comm.Get_rank())
    rows = rng.integers(0, N, size=10)  # any order, repeats
    got = ds.read_rows(rows)
    ok = list(got) == keys
    ok &= all(same_values(got[k], stacked(src, rows, k)) for k in keys)
    sub = ds.read_rows(rows[:3], fields=keys[-1:])
    ok &= list(sub) == keys[-1:] and same_values(sub[keys[-1]], stacked(src, rows[:3], keys[-1]))
    with pytest.raises(KeyError):
        ds.read_rows(rows, fields=["nope"])

    bufs = ds.alloc(16)
    miss0 = None
    if os.environ.get("DDSTORE_PROFILE", "0") not in ("", "0") and method != 0:
        miss0 = [ds.ddstore.get_profile(v)["mr_miss"] for v in ds._var]
    for it in range(3):  # the same buffers, different rows each time
        r = rng.integers(0, N, size=12)
        got = ds.read_rows(r, out=bufs)
        ok &= all(same_values(got[k], stacked(src, r, k)) for k in keys)
        ok &= all(shares(got[k], bufs[k]) for k in keys)
        batch = ds.__getitems__(r[:5], out=bufs)
        ok &= all(same(b, src[int(i)]) for b, i in zip(batch, r[:5]))
    if miss0 is not None:
        ok &= [ds.ddstore.get_profile(v)["mr_miss"] for v in ds._var] == miss0
    with pytest.raises(ValueError):
        ds.read_rows(np.arange(17) % N, out=bufs)  # more rows than the buffers
    ds.release(bufs)
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("loader", ["DataLoader", "ThreadDataLoader"])
def test_windowed_dataset(comm, monkeypatch, method, loader):
    """Windows with stride and dilation, explicit starts, a field subset, and
    whole batches through a loader (one read_rows per batch)."""
    src = TupleSource()
    ds = make(comm, monkeypatch, src, method)
    nf = len(ds._var)

    def window(rows):
        return tuple(stacked(src, rows, j) for j in range(nf))

    wd = WindowedDataset(ds, window=3, stride=2, dilation=2)  # rows s, s+2, s+4
    ok = len(wd) == (N - 5) // 2 + 1
    ok &= same(wd[4], window([8, 10, 12])) and same(wd[-1], window([32, 34, 36]))
    cls = DataLoader if loader == "DataLoader" else ThreadDataLoader
    kw = {} if cls is DataLoader else {"num_workers": 1 if method == 0 else 2}
    got = list(cls(wd, batch_size=4, **kw))
    ref = [
        torch.utils.data.default_collate([window([s, s + 2, s + 4]) for s in range(b, min(b + 8, len(wd) * 2), 2)])
        for b in range(0, len(wd) * 2, 8)
    ]
    ok &= len(got) == len(ref) and all(same(g, r) for g, r in zip(got, ref))

    starts = [0, 10, 30]  # e.g. one window per trajectory
    ws = WindowedDataset(ds, window=2, starts=starts, fields=[0, 2])
    ok &= len(ws) == 3
    w = ws[1]
    ok &= list(w) == [0, 2] and same_values(w[2], stacked(src, [10, 11], 2))
    with pytest.raises(IndexError):
        WindowedDataset(ds, window=2, starts=[N - 1])
    finish(comm, ds)
    assert all_ok(comm, ok)


class Offset(Dataset):
    def __init__(self, n, base):
        self.n, self.base = n, base

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.full((3,), self.base + i, dtype=np.int64)


def test_row_of_concat(comm, monkeypatch):
    """Several sources in one store: row_of maps (source, index) to the row."""
    parts = [Offset(11, 0), Offset(7, 1000), Offset(19, 2000)]
    concat = torch.utils.data.ConcatDataset(parts)
    ds = make(comm, monkeypatch, concat, 0)
    ok = row_of(concat, 0, 3) == 3 and row_of(concat, 2, 0) == 18
    rows = [row_of(concat, s, i) for s, i in [(1, 6), (2, 18), (0, 0)]]
    got = ds.read_rows(rows)[0]
    ok &= same_values(got, np.stack([parts[1][6], parts[2][18], parts[0][0]]))
    for bad in [(3, 0), (1, 7), (0, -1)]:
        with pytest.raises(IndexError):
            row_of(concat, *bad)
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.parametrize("method", METHODS)
def test_thread_loader_reuse_buffers(comm, monkeypatch, method):
    """reuse_buffers: two epochs equal the plain source, no registration per
    read (method 1, DDSTORE_PROFILE=1), setups that can't copy are refused,
    and close() unregisters the pool."""
    src = TupleSource()
    ds = make(comm, monkeypatch, src, method)
    nw = 1 if method == 0 else 2
    ref = list(DataLoader(src, batch_size=8))  # last batch is short (37 = 4*8 + 5)
    loader = ThreadDataLoader(ds, batch_size=8, num_workers=nw, reuse_buffers=True)
    ok = len(loader._pool_sets) == nw
    prof = os.environ.get("DDSTORE_PROFILE", "0") not in ("", "0") and method != 0
    miss0 = [ds.ddstore.get_profile(v)["mr_miss"] for v in ds._var] if prof else None
    for _ in range(2):
        got = list(loader)
        ok &= len(got) == len(ref) and all(same(g, r) for g, r in zip(got, ref))
    if prof:
        ok &= [ds.ddstore.get_profile(v)["mr_miss"] for v in ds._var] == miss0
    sets = loader._pool_sets
    loader.close()
    if method != 0:  # unregistered: releasing again raises
        for bufs in sets:
            with pytest.raises(ValueError):
                ds.release(bufs)

    with pytest.raises(ValueError):  # no auto-collation: nothing copies
        ThreadDataLoader(ds, batch_size=None, reuse_buffers=True)
    with pytest.raises(ValueError):  # custom collate not declared as copying
        ThreadDataLoader(ds, batch_size=8, collate_fn=lambda b: b, reuse_buffers=True)
    with pytest.raises(TypeError):  # dataset without alloc()
        ThreadDataLoader(src, batch_size=8, reuse_buffers=True)
    copying = ThreadDataLoader(
        ds, batch_size=8, num_workers=nw, reuse_buffers=True, collate_copies=True,
        collate_fn=lambda b: [tuple(x.clone() if isinstance(x, torch.Tensor) else
                                    np.array(x, copy=True) for x in s) for s in b],
    )
    got = [s for b in copying for s in b]
    ok &= len(got) == N and all(
        same_values(g[0], src[i][0]) and same_values(g[2], src[i][2])
        for i, g in enumerate(got)
    )
    copying.close()
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_per_sample_fallback(comm, monkeypatch):
    monkeypatch.setenv("DDSTORE_BATCH_GET", "0")
    src = TupleSource()
    ds = make(comm, monkeypatch, src, 0)
    idx = list(range(N))[::-3]
    ok = all(same(b, src[i]) for b, i in zip(ds.__getitems__(idx), idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_ddstore_width_groups(comm, monkeypatch):
    if comm.Get_size() < 4:
        pytest.skip("requires at least 4 ranks")
    src = SingleSource()
    ds = make(comm, monkeypatch, src, 0, ddstore_width=2)
    idx = list(range(N))
    ok = all(same(b, src[i]) for b, i in zip(ds.__getitems__(idx), idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


class _Bad(Dataset):
    def __init__(self, kind):
        self.kind = kind

    def __len__(self):
        return N

    def __getitem__(self, i):
        if self.kind == "shape":
            return np.zeros(3 if i % 5 else 4, dtype=np.float32)
        if self.kind == "dtype":
            return torch.zeros(2, dtype=torch.float16)
        if self.kind == "nested":
            return (np.zeros(2), (1, 2))
        return object()


@pytest.mark.parametrize(
    "kind,exc",
    [
        ("shape", ValueError),
        ("dtype", TypeError),
        ("nested", TypeError),
        ("object", TypeError),
    ],
)
@pytest.mark.parametrize("chunk_size", [None, 3])
def test_unsupported_samples_raise(comm, kind, exc, chunk_size):
    with pytest.raises(exc):
        DistDataset(_Bad(kind), "bad", comm, method=0, chunk_size=chunk_size)
    comm.Barrier()


def test_chunk_size_needs_host_storage(comm):
    with pytest.raises(ValueError):
        DistDataset(TupleSource(), "c", comm, method=0, chunk_size=4, add_device="cpu")
    with pytest.raises(ValueError):
        DistDataset(TupleSource(), "c", comm, method=0, chunk_size=0)
    comm.Barrier()


@pytest.mark.skipif(
    not (HAVE_CXI and HAVE_GPU and FABRIC == "cxi"),
    reason="requires the cxi provider and a GPU",
)
def test_gpu_device_and_add_device(comm, monkeypatch):
    src = TupleSource()
    ds = make(comm, monkeypatch, src, 1, device="cuda", add_device="cuda")
    idx = list(range(N))[::-1]
    batch = ds.__getitems__(idx)
    ok = batch[0][0].is_cuda and isinstance(
        batch[0][2], np.ndarray
    )  # tensors on GPU, numpy stays host
    ok &= all(same(b, src[i]) for b, i in zip(batch, idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.skipif(
    not (HAVE_CXI and HAVE_GPU and FABRIC == "cxi"),
    reason="requires the cxi provider and a GPU",
)
def test_gpu_reuse_buffers(comm, monkeypatch):
    """reuse_buffers with GPU read buffers: the collate's GPU copy must finish
    before a buffer is refilled (several epochs, 2 threads, small pool)."""
    src = TupleSource()
    ds = make(comm, monkeypatch, src, 1, device="cuda")
    ref = list(DataLoader(src, batch_size=4))
    loader = ThreadDataLoader(ds, batch_size=4, num_workers=2, reuse_buffers=True)
    ok = True
    for _ in range(3):
        got = list(loader)
        ok &= got[0][0].is_cuda and len(got) == len(ref)
        ok &= all(same(g, r) for g, r in zip(got, ref))
    loader.close()
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")
def test_method2_reader(comm, monkeypatch, tmp_path):
    monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    hs = comm.bcast(str(tmp_path / "hs") if comm.Get_rank() == 0 else None, root=0)
    src = RecordSource("dict")  # records + tensors: layout round-trips via meta.json
    core = DistDataset(src, "rd", comm, method=2, handshake_dir=hs)
    comm.Barrier()
    ok = True
    if comm.Get_rank() == 0:
        reader = DistDatasetReader("rd", handshake_dir=hs, n_core=comm.Get_size())
        idx = list(range(N))
        ok = len(reader) == N and reader.shapes == core.shapes
        ok &= all(same(b, src[i]) for b, i in zip(reader.__getitems__(idx), idx))
        reader.ddstore.free()
    finish(comm, core)
    assert all_ok(comm, ok)


@pytest.mark.parametrize("chunk_size", [None, 4])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("form", ["void", "array", "recarray", "nested", "dict"])
def test_record_items(comm, monkeypatch, method, form, chunk_size):
    """numpy structured records as items (or a field) come back as the same
    kind of object with the same layout and values."""
    src = RecordSource(form)
    ds = make(comm, monkeypatch, src, method, chunk_size=chunk_size)
    idx = list(range(N))[::-2]
    ok = all(same(ds[i], src[i]) for i in idx[:4])
    ok &= all(same(b, src[i]) for b, i in zip(ds.__getitems__(idx), idx))
    # records don't collate with default_collate: pass collate_fn through
    batches = list(
        ThreadDataLoader(ds, batch_size=5, num_workers=1, collate_fn=lambda b: b)
    )
    ok &= all(
        same(b, src[i]) for i, b in zip(range(N), [x for bt in batches for x in bt])
    )
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_record_dtypes_property(comm, monkeypatch):
    ds = make(comm, monkeypatch, RecordSource("dict"), 0)
    ok = (
        ds.dtypes["rec"] == REC
        and ds.dtypes["t"] == "float32"
        and ds.shapes["rec"] == ()
    )
    finish(comm, ds)
    assert all_ok(comm, ok)
