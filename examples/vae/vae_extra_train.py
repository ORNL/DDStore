"""
VAE example — extra (read-only) group.

Trains the VAE model using data it never loads itself: every batch is read
over RDMA from a core group (see vae_core_server.py) via DDStore method=2's
file-based handshake. The extra ranks form their own DDP process group
among themselves, entirely separate from the core group's communicator.

Usage:
  srun -n<n_extra> python examples/vae/vae_extra_train.py \\
      --handshake-dir ddstore_hs --n-core 4 --epochs 10

Environment (used as defaults if the matching --flag is not given):
  DDSTORE_HANDSHAKE_DIR       shared handshake directory
  DDSTORE_N_CORE              number of core ranks that published the data
  DDSTORE_HANDSHAKE_TIMEOUT_S poll timeout in seconds (default: 300)
  DDSTORE_NIC_MAP             optional precomputed CPU->NIC map (see
                               cpu_nic_map.py --env) for FABRIC_IFACE
                               auto-selection; falls back to a live
                               hwloc-calc query if unset
"""

from __future__ import print_function

import argparse
import os

## torch (and the RCCL/HIP shared libraries it pulls in) must finish loading
## before mpi4py triggers MPI_Init, or their static destructors run in the
## wrong order at interpreter exit and corrupt the heap.
## Do not reorder these imports.
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

from ddp_utils import setup_ddp, get_local_rank
from distdataset import DistDatasetReader
from vae_model import VAE, loss_function

parser = argparse.ArgumentParser(description="VAE MNIST Example - extra (reader) group")
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
    "--handshake-dir",
    type=str,
    default=os.environ.get("DDSTORE_HANDSHAKE_DIR", "./ddstore_hs"),
    help="shared directory published by vae_core_server.py",
)
parser.add_argument(
    "--n-core",
    type=int,
    default=int(os.environ.get("DDSTORE_N_CORE", "4")),
    help="number of core ranks that published the data",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

comm_size, rank = setup_ddp()
local_rank = get_local_rank(rank)

if args.cuda:
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    if torch.xpu.device_count() > 1:
        torch.xpu.set_device(localrank)
        device = torch.device(f"xpu:{local_rank}")
    else:
        device = torch.device("xpu")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("DDP setup:", comm_size, rank, device)

model = VAE().to(device)
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

kwargs = {}

trainset = DistDatasetReader("train", args.handshake_dir, args.n_core)
sampler = torch.utils.data.distributed.DistributedSampler(trainset)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, **kwargs, sampler=sampler
)

testset = datasets.MNIST(
    "data", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, **kwargs
)


def train(epoch):
    model.train()
    train_loss = 0
    train_loader.dataset.ddstore.epoch_begin()
    for batch_idx, (data, _) in enumerate(train_loader):
        train_loader.dataset.ddstore.epoch_end()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
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

    train_loader.dataset.ddstore.epoch_end()
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
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
                    "results/extra_reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.module.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                "results/extra_sample_" + str(epoch) + ".png",
            )

    if rank == 0:
        sentinel = os.path.join(args.handshake_dir, "done_extra")
        with open(sentinel, "w") as f:
            f.write("done\n")
        print("[extra] sentinel written, exiting", flush=True)

    dist.destroy_process_group()
