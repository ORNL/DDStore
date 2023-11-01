import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import sys
import argparse
from tqdm import tqdm
from mpi4py import MPI

from torchvision import datasets, transforms
from distdataset import DistDataset

import time
import numpy as np

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

    trainset = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    trainset = DistDataset(trainset, "trainset", comm, **opt)
    print("trainset size: %d" % len(trainset))

    comm.Barrier()

    if role == 1:
        t = 0
        for i in range(0, len(trainset), 1000):
            if mode == 1:
                i = 0
            print(">>> [%d] consumer asking ... %d" % (rank, i))
            t0 = time.time()
            trainset.__getitem__(i)
            t1 = time.time()
            print(">>> [%d] consumer received: %d (time: %f)" % (rank, i, t1 - t0))
            t += t1 - t0
        print("[%d] consumer done. (avg: %f)" % (rank, t / len(trainset)))
        comm.Barrier()
    else:
        # trainset.ddstore.epoch_begin()
        cnt = 0
        while True:
            if mode == 0:
                i = 0
                print(">>> [%d] producer waiting ..." % (rank))
            else:
                i = np.random.randint(len(trainset))
                print(">>> [%d] producer streaming begin ... %d" % (rank, i))
            rtn = trainset.get(i)
            if mode == 0:
                print(">>> [%d] producer responded." % (rank))
            else:
                print(">>> [%d] producer streaming end." % (rank))
                cnt += 1
                if cnt >= args.epochs * len(trainset):
                    break
            # comm.Barrier()
            """
            if i%500:
                comm.Barrier()
                #trainset.ddstore.epoch_end()
                #trainset.ddstore.epoch_begin()
            """
        # trainset.ddstore.epoch_end()
    sys.exit(0)
