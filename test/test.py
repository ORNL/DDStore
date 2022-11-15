import mpi4py

# mpi4py.rc.thread_level = "serialized"
# mpi4py.rc.threads = False

import numpy as np
from mpi4py import MPI
import argparse
import pyddstore as dds
import sys
import os
import torch
import torch.distributed as dist

import psutil
import socket


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        type=int,
        help="num. of data (default: %(default)s)",
        default=1024 * 1024,
    )
    parser.add_argument("--dim", type=int, help="dim (default: %(default)s)", default=64)
    parser.add_argument("--nbatch", type=int, help="nbatch (default: %(default)s)", default=32)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gloo", help="gloo", action="store_const", dest="backend", const="gloo")
    group.add_argument("--nccl", help="nccl", action="store_const", dest="backend", const="nccl")
    parser.set_defaults(backend="gloo")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = "8889"
    backend = args.backend

    if os.getenv("LSB_HOSTS") is not None:
        ## source: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/Scaling-DL-on-Summit.pdf
        ## The following is Summit specific
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_NODELIST") is not None:
        ## The following is CADES specific
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(comm_size)
    os.environ["RANK"] = str(rank)

    if (backend == "gloo") and ("GLOO_SOCKET_IFNAME" not in os.environ):
        ifname = find_ifname(master_addr)
        if ifname is not None:
            os.environ["GLOO_SOCKET_IFNAME"] = ifname

    dist.init_process_group(backend=backend, init_method="env://")

    ddstore = dds.PyDDStore(comm)

    num = args.num
    dim = args.dim
    nbatch = args.nbatch

    shape = (num, dim)
    dtype = np.float64
    arr = np.ones(shape, dtype=dtype)
    arr = arr * (rank + 1)
    arr = arr.reshape(shape)
    print(rank, "arr", np.mean(arr), arr.nbytes / 1024 / 1024 / 1024, "(GB)")
    ddstore.add("var", arr)
    ddstore.add("var2", arr)

    comm.Barrier()
    idx_list = list()
    buff_list = list()
    for i in range(nbatch):
        ddstore.epoch_begin()
        idx = np.random.randint(num * comm_size)
        buff = np.zeros((1, dim), dtype=dtype)
        ddstore.get("var", buff, idx)
        idx2 = np.random.randint(num * comm_size)
        buff2 = np.zeros((1, dim), dtype=dtype)
        ddstore.get("var", buff2, idx)
        ddstore.epoch_end()
        idx_list.append(idx)
        buff_list.append(buff)
        x = torch.zeros(1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    for i, idx in enumerate(idx_list):
        expected = idx // num + 1
        assert np.mean(buff_list[i]) == expected, (np.mean(buff_list[i]), expected)
    comm.Barrier()
    print(rank, "done.")
    ddstore.free()
    sys.exit(0)
