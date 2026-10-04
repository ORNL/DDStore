"""Microbenchmark for DDStore get(): per-call latency and throughput vs row size.

Each rank adds a shard of random float32 rows, then issues --nget single-row
get()s of uniformly random global rows (most of them remote) and reports the
average per-get latency and per-rank throughput, plus the DDSTORE_PROFILE
breakdown (lock wait / MR / fi_read post / CQ wait / GPU sync).

Run (method 1, cxi), e.g. on 2 nodes:
  DDSTORE_PROFILE=1 DDSTORE_FABRIC=cxi srun -N2 -n16 -c7 --gpus-per-task=1 \\
      python examples/scripts/bench_get.py --row-floats 784,3136 --dest host,gpu

Destinations: "host" reads into a reused numpy row; "gpu" allocates a fresh
torch.empty() per call, like DistDataset.get() with --gpu-dest; "gpu-reuse"
reuses one GPU row (no allocator churn).
"""

import argparse
import os
import threading
import time

## torch must load before mpi4py triggers MPI_Init (see vae-ddp.py).
import torch
import numpy as np
from mpi4py import MPI

import pyddstore as dds

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--row-floats", default="784,3136",
                    help="comma-separated row widths in float32 (784 = MNIST "
                    "28x28 / --image-scale 1, 3136 = 56x56 / --image-scale 2)")
parser.add_argument("--rows-per-rank", type=int, default=4096)
parser.add_argument("--nget", type=int, default=4000,
                    help="get() calls per rank per configuration")
parser.add_argument("--dest", default="host,gpu",
                    help="comma-separated: host, gpu, gpu-reuse")
parser.add_argument("--gpu-source", action="store_true",
                    help="add() the shard as a GPU tensor instead of numpy")
parser.add_argument("--threads", default="1",
                    help="comma-separated thread counts issuing get()s concurrently")
parser.add_argument("--method", type=int, default=int(os.environ.get("DDSTORE_METHOD", "1")))
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
ngpu = torch.cuda.device_count()
device = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0)) % ngpu}") if ngpu else None
if device is not None:
    torch.cuda.set_device(device)

row_floats = [int(x) for x in args.row_floats.split(",")]
dests = args.dest.split(",")
thread_counts = [int(x) for x in args.threads.split(",")]
total_rows = args.rows_per_rank * size

if rank == 0:
    print(f"ranks={size} rows/rank={args.rows_per_rank} nget/rank={args.nget} "
          f"method={args.method} fabric={os.environ.get('DDSTORE_FABRIC', 'hsn')} "
          f"gpu_source={args.gpu_source} profile={os.environ.get('DDSTORE_PROFILE', '0')}",
          flush=True)
    print(f"{'row_B':>8} {'dest':>9} {'thr':>3} {'us/get':>8} {'MB/s/rank':>9} | "
          f"{'sync':>6} {'lock':>6} {'mr':>6} {'miss%':>6} {'read':>6} {'cq':>6} {'other':>6}  (us/get)",
          flush=True)

for nf in row_floats:
    for dest in dests:
        if dest != "host" and device is None:
            continue
        for nthr in thread_counts:
            store = dds.PyDDStore(comm, method=args.method)
            rng = np.random.default_rng(rank)
            shard = np.full((args.rows_per_rank, nf), float(rank), dtype=np.float32)
            if args.gpu_source:
                shard = torch.from_numpy(shard).to(device)
            store.add("x", shard)
            comm.Barrier()
            store.epoch_begin()

            idx = rng.integers(0, total_rows, size=args.nget)
            chunks = np.array_split(idx, nthr)

            def worker(ids):
                reuse_host = np.empty((1, nf), dtype=np.float32)
                reuse_gpu = (torch.empty((1, nf), dtype=torch.float32, device=device)
                             if device is not None else None)
                for g in ids:
                    if dest == "host":
                        store.get("x", reuse_host, int(g))
                    elif dest == "gpu":
                        out = torch.empty((1, nf), dtype=torch.float32, device=device)
                        store.get("x", out, int(g))
                    else:
                        store.get("x", reuse_gpu, int(g))

            # One warm-up get so first-call registration isn't timed as typical.
            worker(idx[:1])
            comm.Barrier()
            p0 = store.get_profile("x")
            comm.Barrier()
            t0 = time.perf_counter()
            ths = [threading.Thread(target=worker, args=(c,)) for c in chunks]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            if device is not None:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            p1 = store.get_profile("x")
            store.epoch_end()

            d = {k: p1[k] - p0[k] for k in p1}
            vals = comm.gather((dt, d), root=0)
            if rank == 0:
                n = sum(v[1]["calls"] for v in vals) or 1
                ngets = args.nget * size
                us_get = 1e6 * sum(v[0] for v in vals) / ngets * nthr
                mbps = nf * 4 * args.nget / (sum(v[0] for v in vals) / size) / 1e6
                s = lambda k: 1e6 * sum(v[1][k] for v in vals) / n
                other = s("py_get") - s("py_sync") - s("lock_wait") - s("mr") - s("read") - s("cq")
                miss = 100.0 * sum(v[1]["mr_miss"] for v in vals) / n
                print(f"{nf * 4:>8} {dest:>9} {nthr:>3} {us_get:>8.1f} {mbps:>9.1f} | "
                      f"{s('py_sync'):>6.1f} {s('lock_wait'):>6.1f} {s('mr'):>6.1f} "
                      f"{miss:>6.1f} {s('read'):>6.1f} {s('cq'):>6.1f} {other:>6.1f}",
                      flush=True)
            # Every rank must finish reading before any rank tears down its
            # endpoint (gather() doesn't synchronize non-root ranks).
            comm.Barrier()
            store.free()
            del store
            comm.Barrier()
