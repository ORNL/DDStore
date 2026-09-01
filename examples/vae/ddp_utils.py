import os
import re
import socket

import psutil
import torch
import torch.distributed as dist

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
    else:
        from mpi4py import MPI

        world_size = MPI.COMM_WORLD.Get_size()
        world_rank = MPI.COMM_WORLD.Get_rank()

    ## Fall back to default
    if world_size is None:
        world_size = 1

    return int(world_size), int(world_rank)


def get_local_rank(rank):
    """
    Determine which GPU on the local node this rank should use.
    Falls back to rank % device_count when no launcher-provided local rank
    is available (e.g. plain mpirun without per-rank GPU visibility).
    """
    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") is not None:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif os.getenv("SLURM_LOCALID") is not None:
        return int(os.environ["SLURM_LOCALID"])
    return 0


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
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        backend = "xccl"
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
    else:
        raise RuntimeError("No parallel backends available")

    world_size, world_rank = init_comm_size_and_rank()
    print(f"DDP: Hi from rank {world_rank} of {world_size}.")

    ## Default setting
    master_addr = "127.0.0.1"
    master_port = os.getenv("MASTER_PORT", "2345")

    if os.getenv("LSB_HOSTS") is not None:
        master_addr = os.environ["LSB_HOSTS"].split()[1]
    elif os.getenv("LSB_MCPU_HOSTS") is not None:
        master_addr = os.environ["LSB_MCPU_HOSTS"].split()[2]
    elif os.getenv("SLURM_STEP_NODELIST") is not None:
        master_addr = parse_slurm_nodelist(os.environ["SLURM_STEP_NODELIST"])[0]
    elif os.getenv("SLURM_NODELIST") is not None:
        master_addr = parse_slurm_nodelist(os.environ["SLURM_NODELIST"])[0]
    elif os.getenv("PBS_O_HOST") is not None:
        if os.environ["PBS_O_HOST"][-19:] == "aurora.alcf.anl.gov":
            from mpi4py import MPI

            RANK = MPI.COMM_WORLD.Get_rank()
            MASTER_ADDR = socket.gethostname() if RANK == 0 else None
            MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
            master_addr = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
        else:
            ## The following is CADES specific
            master_addr = parse_slurm_nodelist(os.environ["PBS_O_HOST"])[0]

    try:
        if backend in ["nccl", "gloo", "xccl"]:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(master_port)
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
