## torch (and the RCCL/HIP shared libraries it pulls in) must finish loading
## before mpi4py triggers MPI_Init, or their static destructors run in the
## wrong order at interpreter exit and corrupt the heap.
## Do not reorder these imports.
import argparse
import os
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

from ddp_utils import setup_ddp
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

comm = MPI.COMM_WORLD
comm_size, rank = setup_ddp()

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("DDP setup:", comm_size, rank, device)

if rank == 0:
    os.makedirs("results", exist_ok=True)
comm.Barrier()

model = VAE().to(device)
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# kwargs = {'pin_memory': True} if args.cuda else {}
kwargs = {}

trainset = DistDataset(
    datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
    "train",
    comm,
)
# trainset = datasets.MNIST('data', train=True, download=True,transform=transforms.ToTensor())
comm.Barrier()
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
    if rank == 0:
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
