"""
VAE example — core (data-holding) group.

Loads the MNIST training set, shards it across the ranks of the job
(MPI.COMM_WORLD, or without MPI the ranks Slurm launched), and publishes it via DDStore method=2 (file-based handshake) so a separate
extra group can train against it over RDMA. This script does no training
itself — it just holds the data in memory until the extra group signals it
is done.

Usage:
  srun -n<n_core> python examples/vae/vae_core_server.py [handshake_dir] [--gpu-source] [--replicate=R] [--image-scale=S]

  --gpu-source  Stack this rank's shard directly on the GPU and add() it in
                place (GPUDirect RDMA source, Phase 2), skipping the host
                round-trip. Requires DDSTORE_METHOD=2 (already set below)
                and DDSTORE_FABRIC=cxi, and one visible GPU per rank
                (--gpus-per-task=1, unlike the --gpus-per-task=0 this
                script normally runs with).
  --replicate=R Repeat the MNIST training set R times (default 1), for
                longer epochs with the same per-sample cost. The extra side
                picks up the row count from the published variable.
  --image-scale=S Upscale images to (28*S)x(28*S) (default 1). The extra
                side derives S from the published row width.

Environment:
  DDSTORE_COMM                mpi or file (default: mpi if pyddstore was
                               built with MPI, else file: ranks from
                               SLURM_PROCID/SLURM_NTASKS, collectives through
                               files in handshake_dir)
  DDSTORE_HANDSHAKE_DIR       overrides handshake_dir positional arg
  DDSTORE_HANDSHAKE_TIMEOUT_S poll timeout in seconds (default: 300)
  DDSTORE_NIC_MAP             optional precomputed CPU->NIC map (see
                               cpu_nic_map.py --env) for FABRIC_IFACE
                               auto-selection; falls back to a live
                               hwloc-calc query if unset
"""

import os
import sys
import time

## torch (pulled in below via torchvision/pyddstore.torch) must finish loading
## before mpi4py triggers MPI_Init, or - if GPU/NCCL use is ever added here -
## their static destructors run in the wrong order at interpreter exit and
## corrupt the heap. Do not reorder these imports (mpi4py is imported below).
import torch
from torchvision import datasets, transforms

import pyddstore
from pyddstore.torch import DistDataset
from vae_model import mnist_transform


def _resolve_dir(arg):
    if arg:
        return arg
    return os.environ.get("DDSTORE_HANDSHAKE_DIR", "./ddstore_hs")


gpu_source = "--gpu-source" in sys.argv
replicate = 1
image_scale = 1
for a in sys.argv[1:]:
    if a.startswith("--replicate="):
        replicate = int(a.split("=", 1)[1])
    elif a.startswith("--image-scale="):
        image_scale = int(a.split("=", 1)[1])
positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
hs_dir = _resolve_dir(positional_args[0] if positional_args else "")
os.environ["DDSTORE_METHOD"] = "2"
os.environ["DDSTORE_HANDSHAKE_DIR"] = hs_dir

if os.environ.get("DDSTORE_COMM", "mpi" if pyddstore.HAVE_MPI else "file") == "mpi":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    barrier = comm.Barrier
else:
    # No MPI: DistDataset(comm=None) finds the same ranks; this FileComm is
    # only for the barriers below.
    os.environ["DDSTORE_COMM"] = "file"
    comm = None
    rank, size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    barrier = pyddstore.FileComm(hs_dir, rank, size, "core").barrier

add_device = None
if gpu_source:
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-source requires a visible CUDA/HIP GPU")
    add_device = torch.device("cuda")

if rank == 0:
    os.makedirs(hs_dir, exist_ok=True)
    for fname in os.listdir(hs_dir):
        if fname.endswith((".bin", ".meta.json")) or fname == "done_extra":
            os.remove(os.path.join(hs_dir, fname))
    print(f"[core] handshake_dir={hs_dir}", flush=True)
barrier()

trainset = datasets.MNIST(
    "data", train=True, download=True, transform=mnist_transform(image_scale)
)
dds_trainset = DistDataset(
    torch.utils.data.ConcatDataset([trainset] * replicate),
    "train",
    comm,
    add_device=add_device,
)
barrier()

if rank == 0:
    print(
        "gpu_source:",
        gpu_source,
        "replicate:",
        replicate,
        "image_scale:",
        image_scale,
        flush=True,
    )

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
barrier()

dds_trainset.ddstore.free()

if rank == 0:
    for fname in os.listdir(hs_dir):
        if fname.endswith((".bin", ".meta.json")):
            try:
                os.remove(os.path.join(hs_dir, fname))
            except OSError:
                pass
    try:
        os.remove(sentinel)
    except OSError:
        pass

print(f"[core rank {rank}] done", flush=True)
