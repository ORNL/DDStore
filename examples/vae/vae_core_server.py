"""
VAE example — core (data-holding) group.

Loads the MNIST training set, shards it across the ranks of MPI.COMM_WORLD,
and publishes it via DDStore method=2 (file-based handshake) so a separate
extra group can train against it over RDMA. This script does no training
itself — it just holds the data in memory until the extra group signals it
is done.

Usage:
  srun -n<n_core> python examples/vae/vae_core_server.py [handshake_dir]

Environment:
  DDSTORE_HANDSHAKE_DIR       overrides handshake_dir positional arg
  DDSTORE_HANDSHAKE_TIMEOUT_S poll timeout in seconds (default: 300)
"""

import os
import sys
import time

## torch (pulled in below via torchvision/distdataset) must finish loading
## before mpi4py triggers MPI_Init, or - if GPU/NCCL use is ever added here -
## their static destructors run in the wrong order at interpreter exit and
## corrupt the heap. Do not reorder these imports.
from torchvision import datasets, transforms

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

from distdataset import DistDataset


def _resolve_dir(arg):
    if arg:
        return arg
    return os.environ.get("DDSTORE_HANDSHAKE_DIR", "./ddstore_hs")


hs_dir = _resolve_dir(sys.argv[1] if len(sys.argv) > 1 else "")
os.environ["DDSTORE_METHOD"] = "2"
os.environ["DDSTORE_HANDSHAKE_DIR"] = hs_dir

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    os.makedirs(hs_dir, exist_ok=True)
    for fname in os.listdir(hs_dir):
        if fname.endswith(".bin") or fname == "done_extra":
            os.remove(os.path.join(hs_dir, fname))
    print(f"[core] handshake_dir={hs_dir}", flush=True)
comm.Barrier()

trainset = datasets.MNIST(
    "data", train=True, download=True, transform=transforms.ToTensor()
)
dds_trainset = DistDataset(trainset, "train", comm)
comm.Barrier()

if rank == 0:
    print(
        f"[core] published {len(dds_trainset)} training rows, waiting for extras...",
        flush=True,
    )

sentinel = os.path.join(hs_dir, "done_extra")
timeout_s = int(os.environ.get("DDSTORE_HANDSHAKE_TIMEOUT_S", "300"))
t0 = time.monotonic()

if rank == 0:
    while not os.path.exists(sentinel):
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError("timed out waiting for done_extra sentinel")
        time.sleep(0.5)
    print("[core] sentinel received, shutting down", flush=True)
comm.Barrier()

dds_trainset.ddstore.free()

if rank == 0:
    for fname in os.listdir(hs_dir):
        if fname.endswith(".bin"):
            try:
                os.remove(os.path.join(hs_dir, fname))
            except OSError:
                pass
    try:
        os.remove(sentinel)
    except OSError:
        pass

print(f"[core rank {rank}] done", flush=True)
