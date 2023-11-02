import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import sys
import argparse
from tqdm import tqdm
from mpi4py import MPI

import torch
from torchvision import datasets, transforms
from distdataset import DistDataset

import time
import numpy as np

import torch.distributed as dist
import os
import socket
import psutil
import re


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

    if os.getenv("DDSTORE_BACKEND") is not None:
        backend = os.environ["DDSTORE_BACKEND"]
    elif dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()

    ## Default setting
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "8889")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--width", type=int, help="ddstore width", default=6)
    parser.add_argument("--mq", action="store_true", help="use mq")
    parser.add_argument("--stream", action="store_true", help="use stream mode")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

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

    use_mq = 1 if args.mq else 0  ## 0: false, 1: true
    role = 1 if args.role == "consumer" else 0  ## 0: producer, 1: consumer
    mode = 1 if args.stream else 0  ## 0: mq, 1: stream mq
    opt = {
        "ddstore_width": args.width,
        "use_mq": use_mq,
        "role": role,
        "mode": mode,
    }

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    print("MPI setup:", comm_size, rank)

    setup_ddp()
    device = torch.device("cpu")
    print("DDP setup:", comm_size, rank, device)

    trainset = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    trainset = DistDataset(trainset, "trainset", comm, **opt)
    print("trainset size: %d" % len(trainset))

    sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    sample_list = list()
    for i in sampler:
        sample_list.append(i)

    comm.Barrier()

    if role == 1:
        for k in range(args.epochs):
            t = 0
            for i in sample_list:
                if mode == 1:
                    i = 0
                print(">>> [%d] consumer asking ... %d" % (rank, i))
                t0 = time.time()
                trainset.__getitem__(i)
                t1 = time.time()
                print(">>> [%d] consumer received: %d (time: %f)" % (rank, i, t1 - t0))
                t += t1 - t0
            print("[%d] consumer done. (avg: %f)" % (rank, t / len(trainset)))
            # comm.Barrier()
    else:
        for k in range(args.epochs):
            # trainset.ddstore.epoch_begin()
            for i in sample_list:
                if mode == 0:
                    i = 0
                    print(">>> [%d] producer waiting ..." % (rank))
                else:
                    print(">>> [%d] producer streaming begin ... %d" % (rank, i))
                rtn = trainset.get(i)
                if mode == 0:
                    print(">>> [%d] producer responded." % (rank))
                else:
                    print(">>> [%d] producer streaming end." % (rank))
            # trainset.ddstore.epoch_end()
    sys.exit(0)
