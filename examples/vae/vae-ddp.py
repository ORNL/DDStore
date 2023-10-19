from __future__ import print_function
import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import distdataset
from distdataset import DistDataset

from mpi4py import MPI
import torch.distributed as dist
import os
import socket
import psutil

"""
Functions for DDP on HPC
"""


def init_comm_size_and_rank():
    world_size = None
    world_rank = 0

    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        ## Summit
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif os.getenv("SLURM_NPROCS") and os.getenv("SLURM_PROCID"):
        ## CADES
        world_size = int(os.environ["SLURM_NPROCS"])
        world_rank = int(os.environ["SLURM_PROCID"])

    ## Fall back to default
    if world_size is None:
        world_size = 1

    return int(world_size), int(world_rank)


def find_ifname(myaddr):
    """
    Find socket ifname for a given ip adress. This is for "GLOO" ddp setup.
    Usage example:
        find_ifname("127.0.0.1") will return a network interface name, such as "lo". "lo0", etc.
    """
    ipaddr = socket.gethostbyname(myaddr)
    ifname = None
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.address == ipaddr:
                ifname = nic
                break
        if ifname is not None:
            break

    return ifname


def parse_slurm_nodelist(nodelist):
    """
    Parse SLURM_NODELIST env string to get list of nodes.
    Usage example:
        parse_slurm_nodelist(os.environ["SLURM_NODELIST"])
    Input examples:
        "or-condo-g04"
        "or-condo-g[05,07-08,13]"
        "or-condo-g[05,07-08,13],or-condo-h[01,12]"
    """
    nlist = list()
    for block, _ in re.findall(r"([\w-]+(\[[\d\-,]+\])*)", nodelist):
        m = re.match(r"^(?P<prefix>[\w\-]+)\[(?P<group>.*)\]", block)
        if m is None:
            ## single node
            nlist.append(block)
        else:
            ## multiple nodes
            g = m.groups()
            prefix = g[0]
            for sub in g[1].split(","):
                if "-" in sub:
                    start, end = re.match(r"(\d+)-(\d+)", sub).groups()
                    fmt = "%%0%dd" % (len(start))
                    for i in range(int(start), int(end) + 1):
                        node = prefix + fmt % i
                        nlist.append(node)
                else:
                    node = prefix + sub
                    nlist.append(node)

    return nlist


def setup_ddp():
    """ "Initialize DDP"""

    if os.getenv("HYDRAGNN_BACKEND") is not None:
        backend = os.environ["HYDRAGNN_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = "8889"

    if os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]

    try:
        if backend in ["nccl", "gloo"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(world_rank)

        if (backend == "gloo") and ("GLOO_SOCKET_IFNAME" not in os.environ):
            ifname = find_ifname(master_addr)
            if ifname is not None:
                os.environ["GLOO_SOCKET_IFNAME"] = ifname

        print(
            "Distributed data parallel: %s master at %s:%s"
            % (backend, master_addr, master_port),
        )

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

    except KeyError:
        print("DDP has to be initialized within a job - Running in sequential mode")

    return world_size, world_rank


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
parser.add_argument("--mq", action="store_true", help="use mq")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--producer",
    help="producer",
    action="store_const",
    dest="role",
    const="producer",
)
group.add_argument(
    "--consumer",
    help="consumer",
    action="store_const",
    dest="role",
    const="consumer",
)
parser.set_defaults(role="consumer")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


comm = MPI.COMM_WORLD
comm_size, rank = setup_ddp()
print("DDP setup:", comm_size, rank, device)

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


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    use_ddstore = hasattr(train_loader.dataset, "ddstore") and hasattr(
        train_loader.dataset.ddstore, "epoch_begin"
    )

    model.train()
    train_loss = 0
    if use_ddstore:
        train_loader.dataset.ddstore.epoch_begin()
    for batch_idx, (data, _) in enumerate(train_loader):
        if use_ddstore:
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

        if use_ddstore:
            train_loader.dataset.ddstore.epoch_begin()

    if use_ddstore:
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
                    "results/reconstruction_" + str(epoch) + ".png",
                    nrow=n,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    # print("main", rank)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.module.decode(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28), "results/sample_" + str(epoch) + ".png"
            )
