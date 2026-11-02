# Patterns

## 1. DDP training, MPI build

```python
import torch                                   # torch first, then pyddstore, then mpi4py
import torch.distributed as dist
from pyddstore.torch import DistStore, DistDataset, ThreadDataLoader
from mpi4py import MPI

store = DistStore(MPI.COMM_WORLD, method=1, device=torch.device("cuda", local_rank))
trainset = DistDataset(train_src, "train", store)
valset = DistDataset(val_src, "val", store)
sampler = torch.utils.data.distributed.DistributedSampler(trainset)
loader = ThreadDataLoader(trainset, batch_size=bs, sampler=sampler, num_workers=1)
for epoch in range(n):
    sampler.set_epoch(epoch)
    for x, y in loader:                        # already on the GPU
        ...
store.free()
```

## 2. DDP training without MPI (torch.distributed)

```python
rank, size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
dist.init_process_group("nccl", rank=rank, world_size=size)   # env:// with MASTER_ADDR/MASTER_PORT from the job script
store = DistStore(dist.group.WORLD)            # method 1; records exchanged with all_gather_object
trainset = DistDataset(train_src, "train", store)
```

The same code works in an MPI build with `DDSTORE_COMM=torch`.

## 3. Forked DataLoader workers (methods 1/2)

Use when per-sample Python work (augmentation, decoding) dominates.

```bash
export DDSTORE_FORK_WORKERS=1                  # before libfabric starts
```

```python
store = DistStore(comm, method=1)              # no device=: workers can't use the GPU
trainset = DistDataset(train_src, "train", store)
loader = torch.utils.data.DataLoader(trainset, batch_size=bs, sampler=sampler,
                                     num_workers=4, pin_memory=True,
                                     persistent_workers=True)
for x, y in loader:
    x = x.to(device, non_blocking=True)
```

## 4. Non-tensor fields

```python
LABELS = ["cat", "dog", "owl"]
ds = DistDataset(src, "pets", store,
                 encode=lambda s: {"x": s["x"], "label": LABELS.index(s["label"])},
                 decode=lambda d, i: {**d, "label": LABELS[d["label"]]})
```

Lookups used in `decode` must exist on every rank.

## 5. Large datasets

- `chunk_size=1024` on the store or dataset: lower peak host memory while loading.
- `add_device=device`: keep the share in GPU memory (frees host memory; GPUDirect reads with `DDSTORE_FABRIC=cxi`).
- Loading reads only this rank's indices from `source`, so the source should open files lazily in `__getitem__`, not load everything in `__init__`.

## 6. Core/extra split (method 2)

Core job holds the data; a separate extra job trains on it.

```python
# core
store = DistStore(comm, method=2, handshake_dir=hs)
ds = DistDataset(src, "train", store)
... wait until training is done ...
store.free()

# extra (separate srun step)
from pyddstore.torch import DistDatasetReader
ds = DistDatasetReader("train", handshake_dir=hs, n_core=n_core_ranks)
```

Needs `--network=single_node_vni,job_vni` and the `SLINGSHOT_VNIS` wrapper on Frontier ([hpc.md](hpc.md)). Example: `examples/vae/vae_core_server.py`, `vae_extra_train.py`, `script/job-vae-core-extra.sh`.

## 7. Time windows

```python
from torch.utils.data import ConcatDataset
from pyddstore.torch import WindowedDataset, row_of

files = ConcatDataset([StepsOf(f) for f in paths])          # one sample per time step
frames = DistDataset(files, "frames", store)
starts = [row_of(files, k, t) for k in range(len(paths))
          for t in range(len(files.datasets[k]) - 2 * dt)]  # windows inside each file
pairs = WindowedDataset(frames, window=3, dilation=dt, starts=starts)
```

## Integration checklist

- [ ] Every rank creates the same datasets in the same order.
- [ ] Sampler is `DistributedSampler` (or equivalent) and `set_epoch` is called.
- [ ] With method 0: same number of batches on every rank (`drop_last` or equal shards).
- [ ] `store.free()` at exit, on every rank.
- [ ] Loss matches the original code for the first epochs.
- [ ] Fetch time measured before and after.
