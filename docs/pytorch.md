# PyTorch integration

`pyddstore.torch` (needs PyTorch: `pip install .[torch]` or an existing PyTorch) turns any map-style dataset into a distributed one:

```python
import torch                                        # import torch before MPI starts
from mpi4py import MPI
from pyddstore.torch import DistDataset, ThreadDataLoader

trainset = DistDataset(my_dataset, "train", MPI.COMM_WORLD)   # each rank loads only its share
sampler = torch.utils.data.distributed.DistributedSampler(trainset)
loader = ThreadDataLoader(trainset, batch_size=128, sampler=sampler, num_workers=1)
for x, y in loader:
    ...
```

- **`DistDataset(source, name, comm=None, ddstore_width=None, device=None, add_device=None, method=None, handshake_dir=None, chunk_size=None, encode=None, decode=None, fields=None)`**: each rank loads its contiguous share of `source` (anything with `len()` and `[i]`) into DDStore; every rank can then read every sample. Samples keep the source's structure (a tensor, numpy array or number; a tuple or list of them; or a dict of them), with each field's shape and dtype. Fields must have the same shape and dtype in every sample; supported dtypes are bool, uint8, int32, int64, float32 and float64. A field (or the whole sample) may also be a numpy structured record (`np.void`, a structured `ndarray`, or `np.recarray`) of any field types: it is stored as raw bytes and comes back as the same kind of object with the same layout (`default_collate` can't batch records, so pass a `collate_fn`). `ds.shapes` / `ds.dtypes` describe the fields, `ds.ddstore` is the underlying `PyDDStore`.
  - `method` (default `DDSTORE_METHOD` or 0) picks the backend; `ddstore_width` splits `comm` into independent stores (see [Partitioned usage](backends.md#partitioned--sub-communicator-usage)).
  - `device` puts tensor fields of read samples on a GPU and `add_device` keeps each rank's share there ([GPUDirect](gpudirect.md)).
  - `chunk_size` loads each rank's share that many samples at a time, writing each chunk into the store before reading the next: peak memory is about the share plus one chunk, instead of about three times the share (400 MiB share: 461 vs 1202 MiB). Host storage only (not with `add_device`).
  - `encode` / `decode` / `fields` handle samples that can't be stored as they are (strings, labels, metadata objects, data shared by a group of samples). `encode(sample)` runs on every source sample before it is stored and returns what to store; `decode(stored, index)` runs on every sample read (`ds[i]`, `__getitems__`, so also in loaders) and rebuilds the full sample, e.g. adding constants or looking up tables by index or by a stored id. `decode` runs on the reading rank, so anything it looks up must exist on every rank. `fields=[...]` keeps only those keys (dict samples) or positions (tuple/list samples); it can't be combined with `encode`. `read_rows()` and `WindowedDataset` return stored rows without `decode`. `DistDatasetReader` takes `decode` too.

```python
LABELS = ["cat", "dog", "owl"]
ds = DistDataset(src, "pets", comm,
                 encode=lambda s: {"x": s["x"], "label": LABELS.index(s["label"]), "group": s["group"]},
                 decode=lambda d, i: {**d, "label": LABELS[d["label"]], "meta": GROUP_INFO[d["group"]]})
```

  - **Batched by default**: `__getitems__` reads a whole batch with one [`get_batch()`](api-pyddstore.md#get_batchname-arr-indices) per field, which `DataLoader` and `ThreadDataLoader` call automatically; `DDSTORE_BATCH_GET=0` reads one sample at a time. With `method=0` batched reads are collective, so every rank must iterate the same number of batches from one thread (`DistributedSampler` does).
- **Reading rows and reusing buffers** (`DistDataset` and `DistDatasetReader`):
  - `ds.read_rows(rows, fields=None, out=None)` reads stored rows `rows` (any order, repeats allowed) of the selected fields (default all), one `get_batch()` per field, and returns a dict of values shaped `(len(rows), *field_shape)`. Keys are the sample's: dict keys, tuple/list positions, or `0` for a single value. With `method=0` it is collective, like `get_batch()`.
  - `ds.alloc(n, fields=None)` returns buffers for `n` rows, one per field and keyed the same way, each [registered](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr) once. Pass them as `out=` to `read_rows()` or `ds.__getitems__(idx, out=)`: reads then skip memory registration, and the results are views into the buffers, so you decide when a buffer can be reused. `ds.release(bufs)` unregisters them; `free()` does too.
- **`WindowedDataset(ds, window, stride=1, dilation=1, starts=None, fields=None)`**: samples made of several stored rows of `ds` (time windows, clips, sequences), each stored row held once. Sample `i` is rows `s, s + dilation, …, s + (window - 1)·dilation` with `s = i·stride`, or `s = starts[i]` when `starts` is given; use `starts` to keep only windows that don't cross a trajectory or file boundary. Fields come back stacked, `(window, *field_shape)`, in the structure of `ds`'s samples (a dict when `fields` is given). A batch of windows is one `read_rows()`.
- **`row_of(concat, source, index)`**: with several sources in one store (a `DistDataset` over a `torch.utils.data.ConcatDataset`), the row of sample `index` of source `source`, e.g. to map (file, trajectory, step) to `starts`:

```python
from torch.utils.data import ConcatDataset
from pyddstore.torch import DistDataset, WindowedDataset, row_of

files = ConcatDataset([StepsOf(f) for f in paths])     # one sample per time step
frames = DistDataset(files, "frames", comm, method=1)
starts = [row_of(files, k, t) for k, f in enumerate(paths)       # windows inside each file
          for t in range(0, len(files.datasets[k]) - 2 * dt)]
pairs = WindowedDataset(frames, window=3, dilation=dt, starts=starts)   # (t, t+dt, t+2dt)
```

- **`DistDatasetReader(name, handshake_dir=None, n_core=None, device=None, decode=None)`**: the same dataset read by a separate `method=2` extra job; it learns the fields from a `{name}.meta.json` file the core group writes next to the handshake records.
- **`ThreadDataLoader(dataset, reuse_buffers=False, collate_copies=False, **DataLoader args)`**: a `DataLoader` whose workers are threads, not forked processes, so it is safe with MPI and GPU buffers. Each batch is fetched, collated and optionally pinned in a worker thread; random draws match `DataLoader`'s. As with `DataLoader`, `iter(loader)` returns a separate iterator for one epoch (`list(it)` or `islice(it, …)` after `next(it)` continue the epoch; iterators over one loader are independent), and while the training step holds a batch, `num_workers * prefetch_factor` more are being fetched, so that many plus one are in memory. One or two workers are enough: reads on one variable are serialized by its lock, and one batched read already keeps the network busy.
  - `reuse_buffers=True` reads every batch into one of a fixed pool of `num_workers` buffer sets from `dataset.alloc(batch_size)`, registered once, instead of fresh buffers registered on every read (for large rows, registration can cost more than the transfer). A worker reads and collates its batch, then returns the set, so the collate must copy: it needs `batch_size` and the default `collate_fn`, or `collate_copies=True` to declare that your `collate_fn` copies; other setups raise. `loader.close()` (or deleting the loader) waits for running fetches and unregisters the pool. `DDSTORE_AFFINITY_WIDTH` / `DDSTORE_AFFINITY_OFFSET` pin worker threads to CPUs.

[examples/vae/vae-ddp.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae-ddp.py) trains a VAE with DDP on top of it:

```bash
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1 --gpu-dest --gpu-source
```

- **`--num-workers`** (default 0): `0` uses PyTorch's standard `DataLoader`; `> 0` uses `ThreadDataLoader` (needs `method=1`/`2`).
- **`--gpu-dest` / `--gpu-source`**: `DistDataset`'s `device` / `add_device`.
- **`--replicate R`** repeats the training set R times (longer epochs); **`--image-scale S`** upscales images to (28·S)² so each row is S² larger. Both default to 1, the original example.
- The [method=2 split](backends.md#file-based-handshake-method2) variant is [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py) (a `DistDataset` core group) + [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) (a `DistDatasetReader`), with the same options.
