# PyTorch integration

`pyddstore.torch` (needs PyTorch) turns any map-style dataset into a distributed one:

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

- **`DistDataset(source, name, store=None, *, device, add_device, chunk_size, encode=None, decode=None, fields=None, comm=None, method=None, handshake_dir=None, ddstore_width=None)`**: each rank loads its share of `source` (any `len()` + `[i]` dataset); every rank reads every sample. Samples keep the source's structure (a value, tuple/list or dict) and each field's shape and dtype, which must be fixed across samples. Dtypes: bool, uint8, int32, int64, float32, float64. Numpy structured records (`np.void`, structured `ndarray`, `np.recarray`) are stored as bytes and returned with the same layout; batch them with a custom `collate_fn`. `ds.shapes`, `ds.dtypes`; `ds.store` is its `DistStore`, `ds.ddstore` the `PyDDStore`.
  - `store`: a `DistStore` (below) to add the dataset to; or a communicator (mpi4py, `torch.distributed` group) or `None` for a store of its own, `DistStore(store, method, handshake_dir, ddstore_width)`. `comm=` is the same as a communicator in `store`; `comm`, `method`, `handshake_dir`, `ddstore_width` are not allowed with a `DistStore`.
  - `device`: put tensor fields of read samples on a GPU. `add_device`: keep each rank's share there ([GPUDirect](gpudirect.md)).
  - `chunk_size`: load the share that many samples at a time; peak memory ≈ share + one chunk instead of ≈ 3× share (400 MiB share: 461 vs 1202 MiB). Host storage only.
  - `device`, `add_device`, `chunk_size` default to the store's; pass them to override.
  - `encode(sample)` → what to store; `decode(stored, index)` → full sample on every read (runs on the reading rank, so its lookups must exist everywhere). For strings, labels, metadata, shared per-group data. `fields=[...]` keeps only those keys/positions (not with `encode`). `read_rows()` and `WindowedDataset` skip `decode`.

```python
LABELS = ["cat", "dog", "owl"]
ds = DistDataset(src, "pets", comm,
                 encode=lambda s: {"x": s["x"], "label": LABELS.index(s["label"]), "group": s["group"]},
                 decode=lambda d, i: {**d, "label": LABELS[d["label"]], "meta": GROUP_INFO[d["group"]]})
```

  - **Batched reads** (default): `__getitems__` does one [`get_batch()`](api-pyddstore.md#get_batchname-arr-indices) per field; `DDSTORE_BATCH_GET=0` reads per sample. With `method=0` they are collective: every rank iterates the same number of batches from one thread (`DistributedSampler` does).
- **Rows and reusable buffers** (`DistDataset`, `DistDatasetReader`):
  - `ds.read_rows(rows, fields=None, out=None)`: stored rows (any order, repeats allowed) as a dict of `(len(rows), *field_shape)` values, keyed by dict key, tuple/list position, or `0`. Collective for `method=0`.
  - `ds.alloc(n, fields=None)`: per-field buffers for `n` rows, [registered](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr) once; pass as `out=` to `read_rows()` / `__getitems__()` to skip registration. Results are views into them. `ds.release(bufs)` (or `free()`) unregisters.
- **`WindowedDataset(ds, window, stride=1, dilation=1, starts=None, fields=None)`**: samples of several stored rows (time windows, clips), each row stored once. Sample `i` is rows `s, s+dilation, …, s+(window-1)·dilation`, with `s = i·stride` or `starts[i]` (use `starts` to avoid crossing file boundaries). Fields are stacked `(window, *field_shape)`. A batch is one `read_rows()`.
- **`row_of(concat, source, index)`**: in a `DistDataset` over a `ConcatDataset`, the row of sample `index` of source `source`, e.g. to build `starts`:

```python
from torch.utils.data import ConcatDataset
from pyddstore.torch import DistDataset, WindowedDataset, row_of

files = ConcatDataset([StepsOf(f) for f in paths])     # one sample per time step
frames = DistDataset(files, "frames", comm, method=1)
starts = [row_of(files, k, t) for k, f in enumerate(paths)       # windows inside each file
          for t in range(0, len(files.datasets[k]) - 2 * dt)]
pairs = WindowedDataset(frames, window=3, dilation=dt, starts=starts)   # (t, t+dt, t+2dt)
```

- **`DistStore(comm=None, method=None, handshake_dir=None, ddstore_width=None, device=None, add_device=None, chunk_size=None)`**: the ranks, backend and handshake of one or more datasets, which share one store. Pass it as `DistDataset`'s `store`.
  - `comm`: mpi4py communicator, `torch.distributed` group, or `None` ([Without MPI](#without-mpi)).
  - `method`: backend (default `DDSTORE_METHOD`, else 0 with MPI, 1 without). `handshake_dir`: for `method=2` and the file handshake. `ddstore_width`: split `comm` into independent stores ([Partitioned usage](backends.md#partitioned--sub-communicator-usage)).
  - `device`, `add_device`, `chunk_size`: defaults for its datasets.
  - Variables of all its datasets live in one `PyDDStore`; a forked worker or `ThreadDataLoader` thread opens one reader for all of them.
  - Dataset names must differ (`store.names`). With `method=2`, they must also differ from other live stores' in the same `handshake_dir`, whose `{name}` files they would overwrite. A `DistDataset` with its own store gets the same check.
  - `store.free()` frees every dataset. Every rank creates the datasets in the same order.

```python
from pyddstore.torch import DistStore, DistDataset

store = DistStore(comm, method=2, handshake_dir=hs, device=dev, add_device=dev)
trainset = DistDataset(ConcatDataset([mnist_train] * r), "train", store)
valset   = DistDataset(ConcatDataset([mnist_val]   * r), "val",   store, add_device=None)  # host shard
testset  = DistDataset(ConcatDataset([mnist_test]  * r), "test",  store)
...
store.free()
```
- **`DistDatasetReader(name, handshake_dir=None, n_core=None, device=None, decode=None)`**: the same dataset from a separate `method=2` extra job; fields come from the core group's `{name}.meta.json`.
- **`ThreadDataLoader(dataset, reuse_buffers=False, collate_copies=False, **DataLoader args)`**: `DataLoader` with worker threads instead of processes; safe with MPI and GPU buffers. Fetch, collate and pinning run in the workers; random draws match `DataLoader`'s. `iter(loader)` gives an independent per-epoch iterator; `num_workers * prefetch_factor` batches are prefetched.
  - **Parallel reads** (methods 1, 2): each worker thread has its own reader (one endpoint per variable). `DDSTORE_THREAD_READERS=0`: threads share the store.
  - **Direct batch reads**: with the default `collate_fn`, tensor, array and number fields are read straight into the batch tensors (same result as `default_collate`). Records, `decode`, custom `collate_fn` and wrapped datasets use the per-sample path, as does `DDSTORE_FAST_COLLATE=0`.
  - **Workers**: one or two are usually enough with batched reads; more help only when preparing a batch costs about a training step ([Measurements](results.md#loaders-with-batched-reads-mpi-optional-branch)).
  - **Profiling**: `dataset.get_profile(name)` sums [counters](api-pyddstore.md#get_profilename) over the store and the threads' readers.
  - `reuse_buffers=True`: one `dataset.alloc(batch_size)` buffer set per worker, registered once (registration can cost more than the transfer for large rows). Sets are reused, so the collate must copy: requires `batch_size` with the default `collate_fn`, or `collate_copies=True`. `loader.close()` waits for fetches and unregisters. `DDSTORE_AFFINITY_WIDTH` / `DDSTORE_AFFINITY_OFFSET` pin workers to CPUs.

## Forked `DataLoader` workers

PyTorch's `DataLoader` with forked workers reads `DistDataset` / `DistDatasetReader` with methods 1 and 2. Set `DDSTORE_FORK_WORKERS=1` before libfabric starts:

```bash
export DDSTORE_FORK_WORKERS=1        # in the job script, or os.environ before importing pyddstore
srun -N2 -n16 --gpus-per-task=1 python train.py
```

```python
import torch
from pyddstore.torch import DistDataset

trainset = DistDataset(my_dataset, "train", handshake_dir=hs)            # method 1 without MPI
loader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=sampler,
                                     num_workers=4, pin_memory=True,
                                     persistent_workers=True)
for x, y in loader:
    x = x.to(device, non_blocking=True)
```

- **How it works**: each worker opens its own reader (`PyDDStore.fork_reader()`) on first read; the inherited store is not used.
- **Shard memory is not copied into workers**: methods 1/2 keep each rank's shard in its own mapping marked `MADV_DONTFORK`. The shard is registered with the NIC, and Linux copies registered (pinned) pages into a child at `fork()`; without this every worker got a private copy of its rank's shard. A child has no shard memory and reads it over RDMA like any other rank. `DDSTORE_DONTFORK=0` turns this off.
- **Host memory only**: a forked process can't use CUDA/HIP, so `device=` is ignored and samples are returned on the CPU; move batches in the training loop (`pin_memory=True` helps). Data stored on the GPU (`add_device=`) is still readable.

  | GPUDirect | forked `DataLoader` worker | `ThreadDataLoader` thread |
  |---|---|---|
  | read from remote GPU memory (`add_device=`, VAE `--gpu-source`) | yes | yes |
  | read into local GPU memory (`device=`, VAE `--gpu-dest`) | no (host) | yes |

  For GPU destinations use `ThreadDataLoader` (`spawn`/`forkserver` workers are not supported with MPI on Frontier).
- **Use `persistent_workers=True`**: new workers every epoch (the default) added up to 0.9 s per epoch in the VAE example; persistent workers matched `ThreadDataLoader` ([Measurements](results.md#loaders-with-batched-reads-mpi-optional-branch)).
- **`method=0`**: not readable from workers (no MPI after fork); use `ThreadDataLoader`.
- **Libfabric settings** (Slingshot/CXI), read when libfabric starts: with MPI, import pyddstore before mpi4py, or export them in the job script.

  | Setting | Set by | Purpose |
  |---|---|---|
  | `FI_MR_CACHE_MONITOR=memhooks` | `DDSTORE_FORK_WORKERS=1`, when the variable is unset or `kdreg2` | Frontier's default `kdreg2` ([OLCF, 2025-05-13](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#id30)) doesn't work in forked processes; `memhooks` does, at the same speed. Without the setting, workers raise an error. Perlmutter leaves the variable unset; there the default, `memhooks`, `userfaultfd` and `disabled` all work. |
  | `FI_CXI_DISABLE_HOST_REGISTER=1` | `pyddstore` on import, unless already set | Skips GPU-runtime (ROCm, CUDA) registration of CXI host buffers, which fails after fork. Perlmutter's default. |

  `FI_MR_CACHE_MONITOR=disabled` also works, with 2–10× slower host reads.
- **Tested** on Frontier and Perlmutter, both builds: methods 1 and 2, 2–8 workers, GPU parents, VAE over MPI, `torch.distributed` and file communicators.
- **Which loader**: `ThreadDataLoader` reads into GPU memory and waits less for batches; forked workers suit heavy per-sample Python work.

## Without MPI

Methods 1 and 2 need no MPI. Pass a `torch.distributed` group as `comm`, or `None`:

| `comm=None` resolves to | When |
|---|---|
| `MPI.COMM_WORLD` | MPI build (default) |
| the default `torch.distributed` group | no MPI, `torch.distributed` initialized |
| a `FileComm` in `handshake_dir`, rank and size from `srun` (`SLURM_PROCID`/`SLURM_NTASKS`) | otherwise |

`DDSTORE_COMM=mpi|torch|file` forces one. Default method is 1; `method=0` raises. `ddstore_width` works; method 2 needs one group.

- **`torch.distributed`**: stores exchange their records with the group's `all_gather_object`. Method 1 needs no `handshake_dir`; method 2 still publishes `{name}.bin` there for extra jobs. `ddstore_width` on the whole world makes groups with `dist.new_group` (other groups fall back to files). With NCCL the exchange runs on the current GPU, in order with the group's other collectives.
- **`FileComm`**: records go through files in `handshake_dir` (shared filesystem).

```python
import torch
import torch.distributed as dist
from pyddstore.torch import DistDataset, ThreadDataLoader

dist.init_process_group("nccl")                  # DDP as usual, ranks from srun
trainset = DistDataset(my_dataset, "train", comm=dist.group.WORLD)  # method 1, no MPI
```

```bash
srun -N2 -n8 --gpus-per-task=1 python train.py
```

Call `torch.cuda.set_device()` first (NCCL/RCCL `all_gather_object`).

[examples/vae/vae-ddp.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae-ddp.py): VAE with DDP.

```bash
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1 --gpu-dest --gpu-source
DDSTORE_FABRIC=cxi srun -n 4 --gpus-per-task=1 python examples/vae/vae-ddp.py --comm=torch   # no MPI: method 1 over the DDP group
```

- **`--num-workers N`** (default 0): threads or forked processes; `0` reads in the main process. Needs `method=1`/`2`.
- **`--loader torch|thread`**: `DataLoader` or `ThreadDataLoader`; default `thread` if workers > 0. `job-vae-single.sh` sets `DDSTORE_FORK_WORKERS=1` for forked workers.
- **`--gpu-dest` / `--gpu-source`**: `DistStore`'s `device` / `add_device`.
- **`--replicate R`**: repeat the training set R times. **`--image-scale S`**: (28·S)² images, S² larger rows. Default 1.
- **`--comm torch`**: the DDP group instead of MPI (default for `DDSTORE_MPI=0`).
- [method=2 split](backends.md#file-based-handshake-method2): [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py) (core) + [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) (`DistDatasetReader`), same options.
