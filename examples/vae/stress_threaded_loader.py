"""
Diagnostic harness for ThreadDataLoader + DistDataset concurrency
correctness. Kept in the repo intentionally (not deleted) for revisiting
later -- see "ThreadDataLoader with GPU buffers and --num-workers > 1" in
the README's Known Limitations section for the full writeup.

What it does: builds a DistDataset, wraps it in ThreadDataLoader
(--num-workers, --batch-size, --epochs configurable), and checks every
(data, label) batch the loader returns against the known-correct MNIST
value at that global index (the raw torchvision dataset, read directly --
no RDMA involved in computing the expected value). Any mismatch means a
race actually corrupted data; the script prints per-sample MISMATCH lines
plus a final GLOBAL mismatches=N summary and exits nonzero if N > 0.

Use --device=cpu for the host-only path (isolates the libfabric/CQ race
in src/common.cxx from the GPU pool entirely -- confirmed necessary via
the threading.Lock in distdataset.py's get(); disabling it crashed with
"double free or corruption" under concurrent access) and --device=cuda
for the GPU-buffer path (--gpu-dest/--gpu-source equivalent; confirmed to
still corrupt data with --num-workers > 1 even with that lock held and a
correctly-sized pool -- see the README section above for the root cause
and what a complete fix needs).

Example (run inside an active salloc, DDSTORE_FABRIC=cxi, DDSTORE_METHOD=1):
  srun -N1 -n2 -c7 python -u stress_threaded_loader.py --device=cpu --num-workers=8 --epochs=5
  srun -N1 -n2 -c7 --gpus-per-task=1 python -u stress_threaded_loader.py --device=cuda --num-workers=8 --epochs=5
"""

## torch (and the RCCL/HIP shared libraries it pulls in) must finish loading
## before mpi4py triggers MPI_Init, or their static destructors run in the
## wrong order at interpreter exit and corrupt the heap.
## Do not reorder these imports.
import argparse
import sys
import time
import torch
import torch.utils.data
from torchvision import datasets, transforms

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

from distdataset import DistDataset
from ddstore_dataloader import ThreadDataLoader

from ddp_utils import setup_ddp, get_local_rank

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

comm_size, rank = setup_ddp()
comm = MPI.COMM_WORLD
local_rank = get_local_rank(rank)

if args.device == "cuda":
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
    gpu_device = device
else:
    gpu_device = None

raw = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())

# See the matching comment in vae-ddp.py: ThreadDataLoader bounds in-flight
# batches to num_workers * prefetch_factor (default 2); the GPU pool must
# be sized to match or get() hands out a slice that's still live elsewhere.
pool_size = (args.num_workers * 2 + 1) * args.batch_size

trainset = DistDataset(
    raw,
    "train",
    comm,
    device=gpu_device,
    add_device=gpu_device,
    pool_size=pool_size,
)
comm.Barrier()

loader = ThreadDataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

mismatches = 0
total = 0
t0 = time.time()
for epoch in range(args.epochs):
    idx_cursor = 0
    for data, label in loader:
        bs = data.shape[0]
        for b in range(bs):
            idx = idx_cursor + b
            expected_img, expected_label = raw[idx]
            got_label = int(label[b].item())
            total += 1
            if got_label != int(expected_label):
                mismatches += 1
                print(
                    f"[rank {rank}] epoch={epoch} idx={idx} LABEL MISMATCH "
                    f"expected={expected_label} got={got_label}",
                    flush=True,
                )
                continue
            got_img = data[b].detach().cpu()
            if not torch.allclose(got_img, expected_img, atol=1e-5):
                maxdiff = (got_img - expected_img).abs().max().item()
                mismatches += 1
                print(
                    f"[rank {rank}] epoch={epoch} idx={idx} DATA MISMATCH maxdiff={maxdiff}",
                    flush=True,
                )
        idx_cursor += bs
elapsed = time.time() - t0

print(
    f"[rank {rank}] DONE device={args.device} num_workers={args.num_workers} "
    f"batch_size={args.batch_size} epochs={args.epochs} total={total} "
    f"mismatches={mismatches} elapsed={elapsed:.1f}s",
    flush=True,
)

comm.Barrier()
all_mismatches = comm.allreduce(mismatches, op=MPI.SUM)
if rank == 0:
    print(f"GLOBAL mismatches={all_mismatches}", flush=True)
    if all_mismatches > 0:
        sys.exit(1)
