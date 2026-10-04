## torch (and the RCCL/HIP shared libraries it pulls in) must finish loading
## before mpi4py triggers MPI_Init, or their static destructors run in the
## wrong order at interpreter exit and corrupt the heap.
## Do not reorder these imports.
import argparse
import os
import time
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributed as dist

import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False
from mpi4py import MPI

import distdataset
from distdataset import DistDataset
from ddstore_dataloader import ThreadDataLoader

from ddp_utils import setup_ddp, get_local_rank
from vae_model import VAE, loss_function

parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--gpu-dest",
    action="store_true",
    default=False,
    help="Allocate the DDStore get() destination buffer directly on the "
    "training device (GPUDirect RDMA, Phase 1), skipping the "
    "host->device copy. Requires DDSTORE_METHOD in (1, 2) and "
    "DDSTORE_FABRIC=cxi (e.g. DDSTORE_METHOD=2 as in run-vae.sh).",
)
parser.add_argument(
    "--gpu-source",
    action="store_true",
    default=False,
    help="Stack this rank's local shard directly on the training device "
    "and add() it in place (GPUDirect RDMA source, Phase 2), skipping "
    "the host round-trip. Same DDSTORE_METHOD/DDSTORE_FABRIC "
    "requirements as --gpu-dest; independent of it -- use either or "
    "both.",
)
parser.add_argument(
    "--loader",
    choices=["default", "threaded"],
    default="default",
    help="DataLoader implementation for the training set. 'threaded' uses "
    "ThreadDataLoader (examples/vae/ddstore_dataloader.py), a "
    "thread-pool-based loader that allows --num-workers > 1 (the default "
    "loader forks worker processes, which hangs with DDStore above 1). "
    "With --gpu-dest/--gpu-source, --num-workers must stay <= 1 even with "
    "--loader=threaded -- confirmed unsafe above that (silent data "
    "corruption, not a crash; see README Known Limitations). Requires "
    "DDSTORE_METHOD 1 or 2. Default: default.",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    metavar="N",
    help="Number of worker threads (--loader=threaded) or worker processes "
    "(--loader=default). With --loader=default, must stay <= 1 -- "
    "forking worker processes after MPI_Init hangs with DDStore; use "
    "--loader=threaded for real parallelism instead. Also forced to 0 "
    "for --loader=default when --gpu-dest/--gpu-source is set "
    "(fork-safety guard). Default: 1.",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

comm = MPI.COMM_WORLD
comm_size, rank = setup_ddp()
local_rank = get_local_rank(rank)

if args.cuda:
    if torch.cuda.device_count() > 1:
        local_rank = get_local_rank(rank)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    if torch.xpu.device_count() > 1:
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
    else:
        device = torch.device("xpu")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(
    "DDP setup:",
    comm_size,
    rank,
    device,
    "gpu_dest:",
    args.gpu_dest,
    "gpu_source:",
    args.gpu_source,
)

if rank == 0:
    os.makedirs("results", exist_ok=True)
comm.Barrier()

model = VAE().to(device)
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# kwargs = {'pin_memory': True} if args.cuda else {}
# --gpu-dest/--gpu-source return CUDA/HIP tensors from __getitem__/add();
# DataLoader worker processes can't safely own GPU state across a fork, so
# the default (forked-process) loader is forced to num_workers=0 whenever
# either is set -- use --loader=threaded for num_workers > 0 with GPU
# buffers instead. Separately, forking *at all* after MPI_Init is a known
# MPI hazard that hangs with DDStore even without GPU buffers -- fail fast
# instead of hanging silently.
if args.loader == "default" and args.num_workers > 1:
    raise RuntimeError(
        "--num-workers > 1 with --loader=default forks worker processes "
        "after MPI_Init, which hangs with DDStore. Use --num-workers=1, or "
        "--loader=threaded for real parallelism."
    )
# --loader=threaded + (--gpu-dest or --gpu-source) + --num-workers > 1 is
# confirmed unsafe by direct experiment: it silently corrupts data (not a
# crash -- wrong pixel values at a low but nonzero rate). Root cause: the
# GPU buffer pool in distdataset.py hands out slots via a per-SAMPLE
# round-robin index shared across threads; slot write order is determined
# by lock-acquisition order, not batch submission/consumption order, so a
# bounded in-flight *batch count* doesn't actually bound which physical
# slots can get overwritten while unread. A real fix needs get() to accept
# a caller-supplied destination buffer so ThreadDataLoader can allocate one
# dedicated region per in-flight batch (not per sample) -- not done yet.
# num_workers<=1 has no concurrent writers, so it's unaffected.
if args.loader == "threaded" and (args.gpu_dest or args.gpu_source) and args.num_workers > 1:
    raise RuntimeError(
        "--loader=threaded with --gpu-dest/--gpu-source and --num-workers > 1 "
        "is known to silently corrupt data (confirmed by direct experiment -- "
        "see README Known Limitations). Use --num-workers=1 with GPU buffers, "
        "or drop --gpu-dest/--gpu-source for real multi-worker parallelism."
    )
if args.gpu_dest or args.gpu_source:
    kwargs = {}
else:
    kwargs = {"num_workers": args.num_workers} if args.num_workers > 0 else {}

# ThreadDataLoader bounds in-flight batches to num_workers * prefetch_factor
# (default prefetch_factor=2, matching torch's own default); the GPU pool
# must hold at least that many batches' worth of samples or get() hands out
# a slice that's still "live" in an unconsumed batch -- see the pool-sizing
# comment in distdataset.py. +1 batch of headroom.
if args.loader == "threaded":
    pool_size = (args.num_workers * 2 + 1) * args.batch_size
else:
    pool_size = None

trainset = DistDataset(
    datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
    "train",
    comm,
    device=device if args.gpu_dest else None,
    add_device=device if args.gpu_source else None,
    pool_size=pool_size,
)
# trainset = datasets.MNIST('data', train=True, download=True,transform=transforms.ToTensor())
comm.Barrier()
sampler = torch.utils.data.distributed.DistributedSampler(trainset)

if args.loader == "threaded":
    if int(os.environ.get("DDSTORE_METHOD", "0")) == 0:
        raise RuntimeError("--loader=threaded requires DDSTORE_METHOD=1 or 2")
    train_loader = ThreadDataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
    )
else:
    # --num-workers applies here too (forked processes), unless --gpu-dest/
    # --gpu-source forced kwargs back to {} above (fork-safety guard).
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, **kwargs, sampler=sampler
    )

print(
    f"train_loader: {type(train_loader).__name__}, num_workers={train_loader.num_workers}"
)

testset = datasets.MNIST(
    "data", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, **kwargs
)


# VAE_PROFILE=1 splits each epoch's wall time into "fetch" (time spent
# inside the DataLoader producing a batch -- __getitem__/get()/collate) vs
# "compute" (forward/backward/optimizer.step()), to see whether a data-
# loading change (e.g. --gpu-dest/--gpu-source) is actually moving the
# needle relative to the rest of the step, rather than guessing from total
# wall time alone. Off by default -- adds one time.perf_counter() pair per
# batch, negligible but not zero.
PROFILE = os.environ.get("VAE_PROFILE") == "1"


def train(epoch):
    model.train()
    train_loss = 0
    fetch_time = 0.0
    compute_time = 0.0
    train_loader.dataset.ddstore.epoch_begin()
    data_iter = iter(train_loader)
    batch_idx = 0
    while True:
        t0 = time.perf_counter()
        try:
            data, _ = next(data_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()
        train_loader.dataset.ddstore.epoch_end()
        # print(rank, device)
        data = data.to(device)
        # print(rank, "data")
        optimizer.zero_grad()
        # print(rank, "optim")
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        # print(rank, "loss:", loss)
        loss.backward()
        # print(rank, "train_loss")
        train_loss += loss.item()
        # print(rank, "backward")
        optimizer.step()
        # print(rank, "step")
        # Skip epoch 1: CUDA/HIP kernel compilation, MIOpen/cuDNN algo
        # selection, and allocator warmup make it dominated by one-time
        # costs unrelated to steady-state fetch/compute timing.
        if PROFILE and epoch > 1:
            torch.cuda.synchronize(device=device)
            t2 = time.perf_counter()
            fetch_time += t1 - t0
            compute_time += t2 - t1
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

        train_loader.dataset.ddstore.epoch_begin()
        batch_idx += 1

    train_loader.dataset.ddstore.epoch_end()
    if rank == 0:
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )
        if PROFILE and epoch > 1:
            print(
                "[profile] epoch {}: fetch={:.3f}s compute={:.3f}s".format(
                    epoch, fetch_time, compute_time
                ),
                flush=True,
            )


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]
                )
                save_image(
                    comparison.cpu(),
                    "results/reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    # print("main", rank)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if rank == 0:
            test(epoch)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.module.decode(sample).cpu()
                save_image(
                    sample.view(64, 1, 28, 28), "results/sample_" + str(epoch) + ".png"
                )

    dist.destroy_process_group()
