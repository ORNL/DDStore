import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import numpy as np
from mpi4py import MPI
import argparse
import pyddstore as dds
import sys
import io
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        type=int,
        help="num. of data (default: %(default)s)",
        default=1024 * 1024,
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
    parser.set_defaults(role="producer")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    num = args.num
    dim = args.dim
    nbatch = args.nbatch
    use_mq = 1 if args.mq else 0
    role = 1 if args.role == "consumer" else 0

    ddstore = dds.PyDDStore(comm, use_mq=use_mq, role=role)

    shape = (num, dim)
    dtype = np.float64
    arr = np.ones(shape, dtype=dtype) * (rank + 1)
    arr = arr.reshape(shape)
    print(rank, "arr", np.mean(arr), arr.nbytes / 1024 / 1024 / 1024, "(GB)")
    lenlist = np.ones(num, dtype=np.int32)
    ddstore.add("var", arr, lenlist)
    print("Add done.")

    comm.Barrier()
    idx_list = list()
    buff_list = list()
    for i in range(nbatch):
        idx = np.random.randint(num * comm_size)
        buff = np.zeros((1, dim), dtype=dtype)
        ddstore.get_ndarray("var", buff, idx)
        idx_list.append(idx)
        buff_list.append(buff)

    if (use_mq == 0) or ((use_mq == 1) and (role == 1)):
        for i, idx in enumerate(idx_list):
            expected = idx // num + 1
            assert np.mean(buff_list[i]) == expected, (np.mean(buff_list[i]), expected)
        print(rank, "Check done.")
    comm.Barrier()
    print(rank, "Done.")
    ddstore.free()
    sys.exit(0)
