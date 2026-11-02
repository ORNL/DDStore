# Patterns

## GPU per rank

```python
import os
import torch

def local_device():
    # One visible GPU per task (srun --gpus-per-task=1, Frontier): ordinal 0.
    # All of the node's GPUs visible (--gpus-per-node, Perlmutter): the local rank.
    if torch.cuda.device_count() > 1:
        local = os.environ.get("SLURM_LOCALID") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
        device = torch.device("cuda", int(local))
    else:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    return device
```

## 1. DDP training, MPI build

```python
import torch                                   # torch first, then pyddstore, then mpi4py
from pyddstore.torch import DistStore, DistDataset, ThreadDataLoader
from mpi4py import MPI

device = local_device()
store = DistStore(MPI.COMM_WORLD, method=1, device=device)
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
import torch.distributed as dist

rank, size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
device = local_device()                        # before NCCL/RCCL init
dist.init_process_group("nccl", rank=rank, world_size=size)   # env:// with MASTER_ADDR/MASTER_PORT from the job script
store = DistStore(dist.group.WORLD, device=device)   # method 1; records exchanged with all_gather_object
trainset = DistDataset(train_src, "train", store)
```

In an MPI build, pass the group explicitly (as here) or set `DDSTORE_COMM=torch`; `comm=None` would pick MPI.

## 3. Forked DataLoader workers (methods 1/2)

Use when per-sample Python work dominates.

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

## 4. Random augmentation

`DistDataset` stores what `source[i]` returns, once. Build the source with the deterministic transforms only, and apply the random ones after the read.

```python
# before: one dataset, transforms in __getitem__
# train_src = ImageFolder(root, transform=T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]))

base = ImageFolder(root, transform=T.Compose([T.Resize((256, 256)), T.PILToTensor()]))   # fixed shape, uint8
trainset = DistDataset(base, "train", store)
augment = T.Compose([T.RandomCrop(224), T.RandomHorizontalFlip()])
```

Then either:

- **On the batch, in the training loop** (keeps `ThreadDataLoader`'s direct batch reads):

  ```python
  for x, y in loader:
      x = torch.stack([augment(img) for img in x]).float().div_(255)   # per-sample randomness
  ```

- **Per sample, in a wrapper** (works with any loader; `ThreadDataLoader` then reads sample by sample):

  ```python
  class Augmented(torch.utils.data.Dataset):
      def __init__(self, ds, fn): self.ds, self.fn = ds, fn
      def __len__(self): return len(self.ds)
      def __getitem__(self, i):
          x, y = self.ds[i]
          return self.fn(x), y
  loader = ThreadDataLoader(Augmented(trainset, augment), ...)
  ```

Storing `uint8` before normalization also stores 4× less than `float32`. With forked workers, seed per worker (`worker_init_fn`) as usual.

## 5. Non-tensor fields

```python
LABELS = ["cat", "dog", "owl"]
ds = DistDataset(src, "pets", store,
                 encode=lambda s: {"x": s["x"], "label": LABELS.index(s["label"])},
                 decode=lambda d, i: {**d, "label": LABELS[d["label"]]})
```

Lookups used in `decode` must exist on every rank. `decode` runs per sample (no direct batch reads).

## 6. Large datasets

Pick one:
- `chunk_size=1024`: lower peak host memory while loading (≈ share + one chunk instead of ≈ 3× share). Host storage.
- `add_device=device`: keep the share in GPU memory, which frees host memory. Can't be combined with `chunk_size` for tensor fields.

The source should open files lazily in `__getitem__`: each rank reads only its own indices.

## 7. Core/extra split (method 2)

The core job holds the data; a separate extra job trains on it.

```python
# core
store = DistStore(comm, method=2, handshake_dir=hs)
ds = DistDataset(src, "train", store)
... wait until training is done ...
store.free()

# extra (separate srun step)
from pyddstore.torch import DistDatasetReader
ds = DistDatasetReader("train", handshake_dir=hs, n_core=n_core_ranks)
...
ds.ddstore.free()
```

On Frontier this needs `--network=single_node_vni,job_vni` and the `SLINGSHOT_VNIS` wrapper ([hpc.md](hpc.md)). Example: [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py), [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py), [job-vae-core-extra.sh](https://github.com/ORNL/DDStore/blob/main/examples/vae/script/job-vae-core-extra.sh).

## 8. Time windows

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

- [ ] Random or epoch-dependent transforms happen after the read, not in the source.
- [ ] Every rank creates the same datasets in the same order.
- [ ] The sampler is `DistributedSampler` (or equivalent), and `set_epoch` is called.
- [ ] With method 0: same number of batches on every rank (`drop_last` or equal shards).
- [ ] `store.free()` at exit, on every rank.
- [ ] The loss matches the original code for the first epochs.
- [ ] Fetch time is measured before and after.
