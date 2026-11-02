"""Smallest complete DDStore training script: DistStore + DistDataset +
ThreadDataLoader + DDP, random augmentation after the read, and a check
that every rank reads the samples the source produces.

  mpirun -n 2 python examples/minimal/train.py                 # MPI build
  srun -n 8 python examples/minimal/train.py --comm torch      # torch.distributed
  DDSTORE_METHOD=1 srun -n 8 python examples/minimal/train.py  # libfabric
"""

import argparse
import os

import torch  # torch first, then pyddstore, then mpi4py
import torch.distributed as dist
import pyddstore
from pyddstore.torch import DistStore, DistDataset, ThreadDataLoader

p = argparse.ArgumentParser()
p.add_argument("--comm", choices=["mpi", "torch"], default=None)
p.add_argument("--samples", type=int, default=1024)
p.add_argument("--epochs", type=int, default=2)
args = p.parse_args()
args.comm = args.comm or ("mpi" if pyddstore.HAVE_MPI else "torch")


class Synthetic(torch.utils.data.Dataset):
    """Deterministic samples: sample i is the same on every call and rank."""

    def __len__(self):
        return args.samples

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        x = torch.rand(1, 8, 8, generator=g)
        return x, int(x.sum() > 32)


def env_int(*names, default=0):
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


rank = env_int("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK")
size = env_int(
    "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE", default=1
)
if torch.cuda.is_available():
    local = env_int("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK")
    device = torch.device("cuda", local if torch.cuda.device_count() > 1 else 0)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
dist.init_process_group(
    "nccl" if device.type == "cuda" else "gloo", rank=rank, world_size=size
)

if args.comm == "mpi":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
else:
    comm = dist.group.WORLD

source = Synthetic()
# method from DDSTORE_METHOD, else 0 with MPI and 1 otherwise. Samples are read
# into host memory; device=device reads into the GPU (methods 1/2, DDSTORE_FABRIC=cxi).
store = DistStore(comm)
trainset = DistDataset(source, "train", store)  # each rank loads its share once

model = torch.nn.parallel.DistributedDataParallel(
    torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2)).to(device)
)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
sampler = torch.utils.data.distributed.DistributedSampler(trainset, seed=0)
loader = ThreadDataLoader(trainset, batch_size=32, sampler=sampler, num_workers=1)
for epoch in range(args.epochs):
    sampler.set_epoch(epoch)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # random augmentation, after the read
        flip = torch.rand(len(x), device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if rank == 0:
        print(f"epoch {epoch}: loss {loss.item():.4f}", flush=True)

# Every rank reads rows from every shard; they match the source.
rows = list(range(0, len(source), max(1, len(source) // 16)))
got = trainset.read_rows(rows)  # collective with method 0
for k, i in enumerate(rows):
    x, y = source[i]
    assert torch.equal(torch.as_tensor(got[0][k]).cpu(), x), i
    assert int(got[1][k]) == y, i
print(f"rank {rank}: {len(rows)} rows match (method {store.method})", flush=True)

store.free()
dist.destroy_process_group()
