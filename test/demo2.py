import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import numpy as np
from mpi4py import MPI
import argparse
import pyddstore2 as dds
import sys
import io
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        type=int,
        help="num. of data (default: %(default)s)",
        default=1024,
    )
    parser.add_argument(
        "--dim", type=int, help="dim (default: %(default)s)", default=64
    )
    parser.add_argument(
        "--nbatch", type=int, help="nbatch (default: %(default)s)", default=32
    )
    parser.add_argument("--mq", action="store_true", help="use mq")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--default",
        action="store_const",
        help="use default",
        dest="mode",
        const="default",
    )
    group.add_argument(
        "--stream", action="store_const", help="use stream", dest="mode", const="stream"
    )
    group.add_argument(
        "--shmem", action="store_const", help="use shmem", dest="mode", const="shmem"
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
    parser.set_defaults(role="producer", mode="default")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    num = args.num
    dim = args.dim
    nbatch = args.nbatch
    use_mq = 1 if args.mq else 0
    role = 1 if args.role == "consumer" else 0
    mode = 1 if args.mode == "stream" else 0
    mode = 2 if args.mode == "shmem" else mode

    print("use_mq=", use_mq, "role=", role, "mode=", mode)
    ddstore = dds.PyDDStore(comm, use_mq=use_mq, role=role, mode=mode)

    shape = (num, dim)
    dtype = np.float64

    dataset = list()
    for i in range(num):
        x = np.ones(shape, dtype=dtype) * (rank * num + i + 1000)
        dataset.append(x)
    ddstore.add("var", dataset)
    print(
        "Create done: ",
        ddstore.buffer("var").getbuffer().nbytes / 1024 / 1024 / 1024,
        "(GB)",
    )

    comm.Barrier()
    idx_list = list()
    arr_list = list()
    for i in range(nbatch):
        idx = np.random.randint(num * comm_size)
        arr = ddstore.get("var", idx, decoder=lambda x: pickle.loads(x))
        # print(rank, idx)
        idx_list.append(idx)
        arr_list.append(arr)

    if (use_mq == 0) or ((use_mq == 1) and (role == 1)):
        for i, idx in enumerate(idx_list):
            expected = idx + 1000
            assert np.mean(arr_list[i]) == expected, (np.mean(arr_list[i]), expected)
    comm.Barrier()
    print(rank, "Done.")
    ddstore.free()
    sys.exit(0)
