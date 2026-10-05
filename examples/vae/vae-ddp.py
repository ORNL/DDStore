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

from mpi4py import MPI

from pyddstore.torch import DistDataset, ThreadDataLoader

from ddp_utils import setup_ddp, get_local_rank
from vae_model import VAE, loss_function, mnist_transform

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
    "DDSTORE_FABRIC=cxi (e.g. job-vae-single.sh --method=1 --gpudirect).",
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
    "--num-workers",
    type=int,
    default=0,
    metavar="N",
    help="Number of DataLoader workers. 0 uses PyTorch's standard "
    "DataLoader in the main process (no worker processes, no fork). "
    "> 0 switches to ThreadDataLoader "
    "(pyddstore.torch), with that many worker threads "
    "-- forked processes can't safely own GPU state or MPI's live state, "
    "so any --num-workers > 0 goes through threads, never a fork. "
    "Requires DDSTORE_METHOD 1 or 2 when > 0. Default: 0.",
)
parser.add_argument(
    "--replicate",
    type=int,
    default=1,
    metavar="R",
    help="Repeat the MNIST training set this many times (torch ConcatDataset), for longer epochs with the same per-sample cost. Default: 1.",
)
parser.add_argument(
    "--image-scale",
    type=int,
    default=1,
    metavar="S",
    help="Upscale MNIST images to (28*S)x(28*S) (bilinear), so each sample is S^2 times larger; the VAE hidden layer grows to 400*S. Default: 1 (plain 28x28).",
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

side = 28 * args.image_scale
model = VAE(input_dim=side * side, hidden=400 * args.image_scale).to(device)
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mnist_train = datasets.MNIST(
    "data", train=True, download=True, transform=mnist_transform(args.image_scale)
)
trainset = DistDataset(
    torch.utils.data.ConcatDataset([mnist_train] * args.replicate),
    "train",
    comm,
    device=device if args.gpu_dest else None,
    add_device=device if args.gpu_source else None,
)
# trainset = datasets.MNIST('data', train=True, download=True,transform=transforms.ToTensor())
comm.Barrier()
sampler = torch.utils.data.distributed.DistributedSampler(trainset)

if args.num_workers > 0:
    if int(os.environ.get("DDSTORE_METHOD", "0")) == 0:
        raise RuntimeError("--num-workers > 0 requires DDSTORE_METHOD=1 or 2")
    train_loader = ThreadDataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
    )
else:
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, sampler=sampler
    )

print(
    f"train_loader: {type(train_loader).__name__}, num_workers={train_loader.num_workers}"
)

testset = datasets.MNIST(
    "data", train=False, download=True, transform=mnist_transform(args.image_scale)
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False
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
                    [data[:n], recon_batch.view(-1, 1, side, side)[:n]]
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
                    sample.view(64, 1, side, side),
                    "results/sample_" + str(epoch) + ".png",
                )

    # DDSTORE_PROFILE=1: where get() time goes, summed over all ranks and
    # both variables (data + labels), all epochs.
    if os.environ.get("DDSTORE_PROFILE", "0") not in ("", "0"):
        ds = trainset.ddstore
        tot = {}
        for var in ("traindata", "trainlabels"):
            for k, v in ds.get_profile(var).items():
                if not k.startswith("py_"):
                    tot[k] = tot.get(k, 0) + v
        prof = ds.get_profile("traindata")
        for k in ("py_gets", "py_get", "py_sync"):
            tot[k] = prof[k]
        tot = {k: comm.allreduce(v) for k, v in tot.items()}
        if rank == 0:
            n = max(tot["calls"], 1)
            us = lambda x: 1e6 * x / n
            us_row = lambda x: 1e6 * x / max(tot["rows"], 1)
            other = (
                tot["py_get"]
                - tot["py_sync"]
                - tot["lock_wait"]
                - tot["mr"]
                - tot["read"]
                - tot["cq"]
            )
            print(
                "[ddstore-profile] all ranks: calls={} rows={} py_calls={} mr_miss={} ({:.1%})".format(
                    tot["calls"],
                    tot["rows"],
                    tot["py_gets"],
                    tot["mr_miss"],
                    tot["mr_miss"] / n,
                )
            )
            print(
                "[ddstore-profile] per call (us): total={:.1f} sync={:.1f} lock_wait={:.1f} "
                "mr={:.1f} read={:.1f} cq={:.1f} other={:.1f}".format(
                    us(tot["py_get"]),
                    us(tot["py_sync"]),
                    us(tot["lock_wait"]),
                    us(tot["mr"]),
                    us(tot["read"]),
                    us(tot["cq"]),
                    us(other),
                ),
                flush=True,
            )
            print(
                "[ddstore-profile] per row (us): total={:.2f} sync={:.2f} read+cq={:.2f}".format(
                    us_row(tot["py_get"]),
                    us_row(tot["py_sync"]),
                    us_row(tot["read"] + tot["cq"]),
                ),
                flush=True,
            )

    dist.destroy_process_group()
