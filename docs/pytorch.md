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

- **`comm`**: an mpi4py communicator, a `torch.distributed` process group, or `None` (see [Without MPI](#without-mpi) below).
- **`DistDatasetReader(name, handshake_dir=None, n_core=None, device=None, decode=None)`**: the same dataset read by a separate `method=2` extra job; it learns the fields from a `{name}.meta.json` file the core group writes next to the handshake records.
- **`ThreadDataLoader(dataset, reuse_buffers=False, collate_copies=False, **DataLoader args)`**: a `DataLoader` whose workers are threads, not forked processes, so it is safe with MPI and GPU buffers. Each batch is fetched, collated and optionally pinned in a worker thread; random draws match `DataLoader`'s. As with `DataLoader`, `iter(loader)` returns a separate iterator for one epoch (`list(it)` or `islice(it, …)` after `next(it)` continue the epoch; iterators over one loader are independent), and while the training step holds a batch, `num_workers * prefetch_factor` more are being fetched, so that many plus one are in memory. One or two workers are enough: reads on one variable are serialized by its lock, and one batched read already keeps the network busy.
  - `reuse_buffers=True` reads every batch into one of a fixed pool of `num_workers` buffer sets from `dataset.alloc(batch_size)`, registered once, instead of fresh buffers registered on every read (for large rows, registration can cost more than the transfer). A worker reads and collates its batch, then returns the set, so the collate must copy: it needs `batch_size` and the default `collate_fn`, or `collate_copies=True` to declare that your `collate_fn` copies; other setups raise. `loader.close()` (or deleting the loader) waits for running fetches and unregisters the pool. `DDSTORE_AFFINITY_WIDTH` / `DDSTORE_AFFINITY_OFFSET` pin worker threads to CPUs.

## Forked `DataLoader` workers

PyTorch's standard `DataLoader` with `num_workers > 0` (forked worker processes) can read a `DistDataset` or `DistDatasetReader` with `method=1` or `2`. Set `DDSTORE_FORK_WORKERS=1` before libfabric starts:

```bash
export DDSTORE_FORK_WORKERS=1        # in the job script, or os.environ before importing pyddstore
srun -N2 -n16 --gpus-per-task=1 python train.py
```

```python
import torch
from pyddstore.torch import DistDataset

trainset = DistDataset(my_dataset, "train", handshake_dir=hs)            # method 1 without MPI
loader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=sampler,
                                     num_workers=4, pin_memory=True)
for x, y in loader:
    x = x.to(device, non_blocking=True)
```

- **How it works**: a worker never uses the store it inherited from the parent (its endpoint belongs to the parent). On its first read it opens a store of its own (`PyDDStore.fork_reader()`), with its own libfabric endpoint, from the record table it inherited, and reads the same data. The parent's store keeps working.
- **Host memory only**: in a worker, `device=` is ignored and samples come back on the CPU, since a forked process can't use the GPU. Move batches in the training loop (`pin_memory=True` helps). GPUDirect (`device=`/`add_device=`) still applies to reads in the parent.
- **`method=0`** can't be read from a worker (MPI doesn't work after fork): use `ThreadDataLoader`.
- **Libfabric settings** (Slingshot/CXI; read when libfabric starts, so import pyddstore before mpi4py, or export them in the job script):

  | Setting | Who sets it | Why |
  |---|---|---|
  | `FI_MR_CACHE_MONITOR=memhooks` | `DDSTORE_FORK_WORKERS=1`, if the variable is unset or `kdreg2` | Frontier exports `kdreg2` by default ([OLCF system updates, 2025-05-13](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#id30)). Its monitor thread doesn't survive fork, so once the parent has read from the store or used the GPU, a worker's first read hangs. `memhooks` runs in the process. Without `DDSTORE_FORK_WORKERS=1` a worker under `kdreg2` raises an error instead of hanging. |
  | `FI_CXI_DISABLE_HOST_REGISTER=1` | `pyddstore` on import (and DDStore), unless already set | Otherwise the CXI provider registers its host receive buffers with ROCm, which fails after fork, and the worker's endpoint can't start (`fi_enable` −262). |

  On Frontier, `memhooks` measured the same as `kdreg2` (`bench_get`, 16 ranks, host and GPU destinations, single and batched reads). `FI_MR_CACHE_MONITOR=disabled` also works with workers but makes host reads 2× (single row) to 10× (batched) slower.
- **Tested** on Frontier with both builds (MPI and `DDSTORE_MPI=0`): methods 1 and 2, 2 to 8 workers, a parent that reads before forking and uses the GPU (GPUDirect), new workers every epoch, and the VAE example over MPI, `torch.distributed` and file communicators. With MPI, import pyddstore (or `pyddstore.torch`) before mpi4py, or export `FI_MR_CACHE_MONITOR=memhooks` in the job script: MPI starts libfabric when mpi4py is imported.
- **Which loader**: `ThreadDataLoader` avoids the fork and reads straight into GPU memory, and was faster in the VAE example (fetch 0.01 s vs 0.09 s per epoch with 4 workers). Forked workers help when the rest of the pipeline (decoding, augmentation in Python) needs separate processes.

## Without MPI

`DistDataset` needs no MPI when its stores use method 1 or 2. Pass `comm=` a `torch.distributed` process group, or leave it `None`:

| `comm=None` resolves to | When |
|---|---|
| `MPI.COMM_WORLD` | MPI build (default) |
| the default `torch.distributed` group | no MPI, `torch.distributed` initialized |
| a `FileComm` in `handshake_dir`, rank and size from `srun` (`SLURM_PROCID`/`SLURM_NTASKS`) | otherwise |

`DDSTORE_COMM=mpi|torch|file` forces one. Without MPI the stores exchange their records through files in `handshake_dir` (on a filesystem every rank sees), the default method is 1, and `method=0` raises. `ddstore_width` works (each group's stores get their own exchange files); method 2 still needs one group.

```python
import torch
import torch.distributed as dist
from pyddstore.torch import DistDataset, ThreadDataLoader

dist.init_process_group("nccl")                  # DDP as usual, ranks from srun
trainset = DistDataset(my_dataset, "train", comm=dist.group.WORLD,
                       handshake_dir="/lustre/.../hs")    # method 1, no MPI
```

```bash
srun -N2 -n8 --gpus-per-task=1 python train.py
```

`all_gather_object` over NCCL/RCCL needs `torch.cuda.set_device()` first, as for DDP.

[examples/vae/vae-ddp.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae-ddp.py) trains a VAE with DDP on top of it:

```bash
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1 --gpu-dest --gpu-source
DDSTORE_FABRIC=cxi srun -n 4 --gpus-per-task=1 python examples/vae/vae-ddp.py --comm=torch   # no MPI: method 1 over the DDP group
```

- **`--num-workers`** (default 0): worker threads (`ThreadDataLoader`) or forked processes (PyTorch `DataLoader`); `0` reads in the main process.
- **`--loader`** `torch|thread`: PyTorch's standard `DataLoader`, or `ThreadDataLoader`. Workers need `method=1`/`2`. Default: `thread` if `--num-workers > 0`, else `torch`. With `torch` and workers, see [Forked DataLoader workers](#forked-dataloader-workers); `job-vae-single.sh` sets `DDSTORE_FORK_WORKERS=1` for it.
- **`--gpu-dest` / `--gpu-source`**: `DistDataset`'s `device` / `add_device`.
- **`--replicate R`** repeats the training set R times (longer epochs); **`--image-scale S`** upscales images to (28·S)² so each row is S² larger. Both default to 1, the original example.
- **`--comm torch`**: `DistDataset` over the DDP process group instead of MPI (the default for a `DDSTORE_MPI=0` build). `job-vae-single.sh --comm=torch`.
- The [method=2 split](backends.md#file-based-handshake-method2) variant is [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py) (a `DistDataset` core group) + [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) (a `DistDatasetReader`), with the same options.
