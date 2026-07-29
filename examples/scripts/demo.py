import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import numpy as np
from mpi4py import MPI
import argparse
import pyddstore as dds
import sys


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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    ddstore = dds.PyDDStore(comm, method=1)

    num = args.num
    dim = args.dim
    nbatch = args.nbatch

    shape = (num, dim)
    dtype = np.float64
    arr = np.ones(shape, dtype=dtype) * (rank + 1)
    arr = arr.reshape(shape)
    print(rank, "arr", np.mean(arr), arr.nbytes / 1024 / 1024 / 1024, "(GB)")
    ddstore.add("var", arr)

    comm.Barrier()
    idx_list = list()
    buff_list = list()
    for i in range(nbatch):
        ddstore.epoch_begin()
        idx = np.random.randint(num)
        buff = np.zeros((1, dim), dtype=dtype)
        ddstore.get("var", buff, idx)
        ddstore.epoch_end()
        idx_list.append(idx)
        buff_list.append(buff)

    for i, idx in enumerate(idx_list):
        expected = idx // num + 1
        assert np.mean(buff_list[i]) == expected, (np.mean(buff_list[i]), expected)
    comm.Barrier()
    print(rank, "done.")
    ddstore.free()
    sys.exit(0)
