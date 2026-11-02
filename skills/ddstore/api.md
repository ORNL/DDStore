# pyddstore.torch API (2.1)

```python
from pyddstore.torch import (DistStore, DistDataset, DistDatasetReader,
                             ThreadDataLoader, WindowedDataset, row_of)
```

## DistStore

```python
DistStore(comm=None, method=None, handshake_dir=None, ddstore_width=None,
          device=None, add_device=None, chunk_size=None)
```

Ranks, backend and handshake shared by one or more datasets. Collective.

- `comm`: mpi4py communicator, `torch.distributed` group, or `None` (MPI build: `MPI.COMM_WORLD`; otherwise the default torch group if initialized, else files in `handshake_dir`). `DDSTORE_COMM=mpi|torch|file` forces one.
- `method`: 0, 1 or 2; default `DDSTORE_METHOD`, else 0 with an MPI communicator and 1 otherwise (a torch group in an MPI build gets 1). Method 0 needs an MPI communicator.
- `handshake_dir`: shared directory; needed for method 2 and the file communicator. Default `DDSTORE_HANDSHAKE_DIR`, else `./ddstore_hs`.
- `ddstore_width`: split `comm` into independent stores of this many ranks (each rank reads only its group's data).
- `device`, `add_device`, `chunk_size`: defaults for its datasets.
- `store.names`, `store.free()`: frees every dataset of the store and this process's readers. Datasets have no `free()` of their own; use `ds.store.free()`.

## DistDataset

```python
DistDataset(source, name, store=None, ddstore_width=None, device=<store's>,
            add_device=<store's>, method=None, handshake_dir=None,
            chunk_size=<store's>, encode=None, decode=None, fields=None, comm=None)
```

Each rank loads its share of `source`; every rank reads every sample. Collective.

- `source`: map-style dataset. Samples: a value, tuple/list or dict of tensors, arrays or numbers; fixed shape and dtype per field. `source[i]` is called once per sample, here; random transforms in it are frozen.
- `name`: unique within the store.
- `store`: a `DistStore`, or a communicator/`None` for a store of its own (then `method`, `handshake_dir`, `ddstore_width` apply; with a `DistStore` they raise).
- `device`: put read tensor fields on this GPU (GPUDirect RDMA: methods 1/2 with `DDSTORE_FABRIC=cxi`). Main process or `ThreadDataLoader` only; forked workers ignore it. A CPU device is `None`.
- `add_device`: keep this rank's share of tensor fields in GPU memory.
- `chunk_size`: load the share this many samples at a time (peak host memory ≈ share + one chunk instead of ≈ 3× share). Host storage only: raises with `add_device` when a field is a tensor.
- `encode(sample) -> stored`, `decode(stored, index) -> sample`: for strings, labels, metadata. `decode` runs on the reading rank.
- `fields=[...]`: keep only these keys/positions (not with `encode`).
- Attributes and methods: `ds.shapes`, `ds.dtypes`, `ds.store`, `ds.ddstore` (the `PyDDStore`), `ds.read_rows(rows, fields=None, out=None)`, `ds.alloc(n, fields=None)`, `ds.release(bufs)`, `ds.get_profile(name)`.
- Batches: `__getitems__` reads a batch with one `get_batch()` per field. With method 0 this is collective.

## DistDatasetReader

```python
DistDatasetReader(name, handshake_dir=None, n_core=None, device=None, decode=None)
```

Reads a method-2 `DistDataset` from a separate job (the "extra" ranks). `n_core`: number of ranks in the core group (`DDSTORE_N_CORE`). Free with `reader.ddstore.free()`.

## ThreadDataLoader

```python
ThreadDataLoader(dataset, reuse_buffers=False, collate_copies=False, **DataLoader_args)
```

`DataLoader` with worker threads. Works with all methods and GPU destinations. `num_workers=0` reads in the main thread. Call `loader.close()` with `reuse_buffers=True`.

## WindowedDataset, row_of

```python
WindowedDataset(ds, window, stride=1, dilation=1, starts=None, fields=None)
row_of(concat, source, index)
```

Samples made of several stored rows (time windows); each row stored once. `row_of` maps (source index in a `ConcatDataset`, sample index) to a row, to build `starts`.

## Low level: PyDDStore

```python
import pyddstore as dds
store = dds.PyDDStore(comm, method=0)  # also: handshake_dir=, rank=, size= without MPI
store.add(name, ndarray)               # collective; rows are this rank's share
store.get(name, out, start)            # global row index
store.get_batch(name, out, indices)
store.free()
```

`pyddstore.HAVE_MPI` is `False` for a `DDSTORE_MPI=0` build.
