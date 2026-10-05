# DDStore measurements and findings

Measurements behind the recommendations in the [README](../README.md),
collected on the `check-thread` branch in October 2026. Unless noted:
`method=1`, `DDSTORE_FABRIC=cxi`, `vae-ddp.py` with `VAE_PROFILE=1`, epoch
times averaged over all epochs but the first. Epochs are short (0.1–0.6 s),
so treat differences under ~10% as noise; numbers from different jobs vary by
up to ~2× (different nodes), comparisons within one table are from one job.

Machines: **Frontier** (AMD MI250X, ROCm 7.2, 8 ranks × 7 cores per node) and
**Perlmutter** (NVIDIA A100, CUDA 13, 4 ranks × 32 cores per node).

## Batched reads (`get_batch`) vs one `get()` per sample

VAE, seconds per epoch, per-sample (`DDSTORE_BATCH_GET=0`) → batched (`1`).
Losses are identical within every row.

**Frontier, 2 nodes × 8 ranks, 8 epochs** (loss 8.8960 at S=1, 30.0178 at S=2):

| `--image-scale` | host, 0 workers | host, 1 worker | GPU, 0 workers | GPU, 1 worker | GPU, 2 workers |
|---|---|---|---|---|---|
| 1 | 0.247 → 0.132 | 0.156 → 0.126 | 0.263 → 0.135 | 0.259 → 0.127 | 0.403 → 0.134 |
| 2 | 0.395 → 0.286 | 0.293 → 0.265 | 0.428 → 0.277 | 0.416 → 0.264 | 0.518 → 0.297 |

"GPU" = `--gpu-dest --gpu-source`.

**Frontier, 4 nodes × 8 ranks, `job-vae-single.sh` (3 epochs)** (loss 6.7155 / 24.9987):

| | S=1 | S=2 |
|---|---|---|
| method 0, 0 workers | 0.241 → 0.088 | 0.356 → 0.203 |
| method 1, 0 workers | 0.128 → 0.088 | 0.254 → 0.185 |
| method 1, 2 workers | 0.096 → 0.072 | 0.218 → 0.175 |
| method 1, 4 workers | 0.113 → 0.067 | 0.193 → 0.174 |

**Perlmutter, 2 nodes × 4 ranks, 8 epochs** (loss 15.3781 at S=1):

| | per-sample → batched |
|---|---|
| method 1, host, 0 workers | 0.398 → 0.208 |
| method 1, host, 2 workers | 0.380 → 0.181 |
| method 1, GPU, 0 workers | 0.508 → 0.204 |
| method 1, GPU, 2 workers | 0.678 → 0.183 |
| method 0, host, 0 workers | 0.849 → 0.219 |

## Where `get()` time goes (`DDSTORE_PROFILE=1`)

Frontier, 2 nodes × 8 ranks, VAE, µs per call, one `get()` per sample:

| run | total | GPU sync | lock wait | MR | read post | CQ wait | other |
|---|---|---|---|---|---|---|---|
| host, 0/1 workers (S=1) | 7.9 / 9.3 | – | 0.0 | 0.1 | 0.6 | 3.6 | 3.6 / 5.0 |
| GPU, 0 workers (S=1 / S=2) | 13.6 / 13.6 | 3.8 / 3.7 | 0.1 | 0.3 | 0.6 | 3.7 / 4.0 | ~5 |
| GPU, 1 worker (S=1 / S=2) | 142.7 / 198.7 | **131.6 / 187.3** | 0.1 | 0.4 | 0.6 | 3.7 / 4.0 | ~6 |
| GPU, 2 workers (S=1 / S=2) | 310.7 / 452.8 | **287.2 / 429.3** | 0.2 | 0.7 | 0.7 | 3.7 / 4.1 | ~18 |

- The per-call whole-device `torch.cuda.synchronize()` dominates the GPU path
  once a worker thread runs next to training: the worker's sync waits for the
  training kernels. With no workers the GPU is idle and the sync costs ~4 µs.
- Lock contention (≤0.2 µs) and MR registration (<1 µs, even at 100% cache
  misses; libfabric caches registrations) are negligible. The RDMA round trip
  is ~4–5 µs.
- With `get_batch()` the per-row sync drops to ~0.1 µs and compute time
  recovers (training no longer waits behind the workers' syncs).

## `bench_get.py`: µs per row

Single-row `get()` vs `get_batch()` of 128 rows, 1 thread, host vs fresh GPU
destination.

| row | Frontier host, 1 / 128 | Frontier GPU, 1 / 128 | Perlmutter host, 1 / 128 | Perlmutter GPU, 1 / 128 |
|---|---|---|---|---|
| 3 KB | 9.0 / 0.68 | 20.9 / 0.78 | 8.6 / 0.82 | 21.6 / 0.99 |
| 12.5 KB | 10.1 / 1.82 | 21.9 / 1.38 | 9.3 / 1.19 | 22.4 / 1.80 |
| 200 KB | 32.5 / 27.5 | 37.9 / 19.6 | – | – |
| 1 MB | 160 / 189 | 134 / 99 | 88 / 75 | 142 / 142 |

- Batching makes small rows 10–27× cheaper per row on both machines.
- Host vs GPU destination is platform-dependent: on Frontier GPUDirect wins
  from ~12.5 KB rows (up to ~10.6 GB/s per rank); on Perlmutter host
  destinations win at every size (1 MB: ~14 vs ~7.4 GB/s per rank).
- On Frontier, host-destination batches of 1 MB rows are slower than single
  reads (not investigated).
- A second thread adds no per-rank throughput: the per-variable lock
  serializes transfers on one variable (lock wait ≈ transfer time at large
  rows). One batch already keeps up to 128 reads in flight.

## `method=0`: per-row `MPI_Get` vs collective `get_batch`

Frontier, 16 ranks, host, µs per row (per-row `get()` → batch 128, 2 MiB rounds):
3 KB 28.8 → 3.0; 12.5 KB 34.9 → 7.5; 200 KB 170 → 111; 1 MB 739 → 685.

Round size (`DDSTORE_ALLTOALL_MAX_BYTES`) at batch 128: 200 KB rows — no cap
272, 2 MiB 111, 8 MiB 125, 32 MiB 267; 1 MB rows — no cap 1765, 2 MiB 685,
8 MiB 696, 32 MiB 1262. One unbounded exchange of large rows is slower than
per-row reads; 2–8 MiB rounds fix it. One-sided `method=1` batching is still
faster at every size (3 KB: 1.6 µs/row in the same job).

## Worker threads (`ThreadDataLoader`), before batching

Frontier, 2 nodes × 8 ranks, one `get()` per sample, s/epoch (fetch / total):

| `--num-workers` | host | GPU (`--gpu-dest --gpu-source`) |
|---|---|---|
| 0 (`DataLoader`) | 0.11 / 0.22 | 0.15 / 0.25 |
| 1 | 0.02 / 0.14–0.16 | 0.09 / 0.26 |
| 2 | 0.03 / 0.16 | 0.11 / 0.31 |
| 4 | 0.04 / 0.17 | 0.12 / 0.30 |
| 8 | 0.08 / 0.21 | 0.15 / 0.35–0.38 |

One worker hid the fetch on the host path; more workers only contended for
the lock and the GIL. On the GPU path threads did not help (per-call device
sync). Moving collation into the worker thread later brought host, 1 worker
to fetch ≈ 0.006 s / total ≈ 0.15 s. Pinned memory with
`.to(device, non_blocking=True)` gave no gain (and was ~3× slower with 0
workers, where pinning runs on the training thread).

## Correctness experiments

- **GPU sync in `get()`**: removing it made every `--gpu-dest` VAE run abort
  in the first epoch with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` (GPU
  memory fault) on every rank (Frontier); host and `--gpu-source`-only runs
  were unaffected. The sync stays.
- **Per-variable lock**: without it, concurrent `get()` calls crashed with
  `double free or corruption`; it protects the shared recv fields and MR cache.
- **GPU destination-buffer pool** (removed): round-robin slices of one
  pre-registered buffer corrupted data under `ThreadDataLoader` with more than
  one worker (slot order followed lock acquisition, not batch order). Each
  `get()` now uses its own fresh tensor; MR registration is cheap enough.
- **MPI thread level**: only the main thread calls MPI. `vae-ddp.py` and the
  test suites gave identical results and timing with `MPI_THREAD_SINGLE`,
  `FUNNELED` and `MULTIPLE` (Frontier, Cray MPICH).

## HIP streams and hardware queues (Frontier, ROCm 7.2)

HIP maps streams onto `GPU_MAX_HW_QUEUES` hardware queues per GPU per process
(default 4); extra streams share a queue round-robin, and work in a shared
queue runs in order. With the default stream kept busy, 12 of 16 new streams
were independent of it by default (every 4th collided, including the first
created), 14 of 16 with `GPU_MAX_HW_QUEUES=8`, 15 of 16 with `16`. Relevant if
data-loading threads get their own streams.

## Perlmutter validation (2 nodes × 4 A100, CUDA 13, cxi)

One 2-node debug job covering what Frontier could not:

- **Tests**: `test_single` 14/14, `test_multirank` 5/5, `test_get_batch`
  20/20 on 8 ranks, `test_gpu_rdma` 14/14 (CUDA GPUDirect, `FI_HMEM_CUDA`).
- **VAE**: identical losses for every variant (method 0/1, host/GPU, 0/2
  workers, per-sample/batched): 15.3781 at S=1, 54.7914 at S=2.
- **Slingshot**: each step's `SLINGSHOT_VNIS` is `<own VNI>,<job VNI>`, job
  VNI last, as on Frontier, so the same wrapper works. Ranks spread over
  `cxi0`–`cxi3`. Single-node steps work even without `--network` flags (no
  `SLINGSHOT_*` variables; cxi falls back to a default CXI service).
- **core/extra**: split-node passes (host and `--gpu-dest`); colocate works
  only with `job_vni` + the wrapper + `srun --overlap`.
- **DDP setup**: training ranks must see all 4 GPUs of their node
  (`--gpus-per-node=4`, each picks `cuda:$SLURM_LOCALID`); with
  `--gpus-per-task=1`, NCCL 2.29 fails in DDP setup with "Cuda failure 101
  'invalid device ordinal'". The job scripts handle this.
