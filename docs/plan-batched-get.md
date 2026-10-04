# Plan: batched `get` (`get_batch`)

Status: planned, not started (updated 2026-10-04, branch `check-thread`, after
`3dadb32`, which added `DDSTORE_PROFILE`, `bench_get.py` and the VAE
`--replicate`/`--image-scale` options used below).

## Motivation (measured)

Today `DistDataset.get()` makes one `PyDDStore.get()` per sample, and each call
does: allocate a 1-row tensor, whole-device `torch.cuda.synchronize()` (GPU
destination only), take the per-variable `recv_lock`, check/register the
recv MR, post one `fi_read`, busy-poll the CQ, return to Python.

`DDSTORE_PROFILE=1` on Frontier (`method=1`, cxi, 2 nodes x 8 ranks), µs per
`get()`:

| VAE run | total | GPU sync | lock wait | MR | read post | CQ wait | other |
|---|---|---|---|---|---|---|---|
| host, 0/1 workers (S=1) | 7.9 / 9.3 | – | 0.0 | 0.1 | 0.6 | 3.6 | 3.6 / 5.0 |
| GPU, 0 workers (S=1 / S=2) | 13.6 / 13.6 | 3.8 / 3.7 | 0.1 | 0.3 | 0.6 | 3.7 / 4.0 | ~5 |
| GPU, 1 worker (S=1 / S=2) | 142.7 / 198.7 | **131.6 / 187.3** | 0.1 | 0.4 | 0.6 | 3.7 / 4.0 | ~6 |
| GPU, 2 workers (S=1 / S=2) | 310.7 / 452.8 | **287.2 / 429.3** | 0.2 | 0.7 | 0.7 | 3.7 / 4.1 | ~18 |

`bench_get.py`, single-row `get()`, 1 thread, µs: 3 KB host 8.6 / GPU 19.5
(12.2 reusing the buffer); 12.5 KB host 9.9 / GPU 20.5; 200 KB host 53 / GPU 37;
1 MB host 206–276 / GPU 134. A second thread adds no per-rank throughput: lock
wait ≈ transfer time at large rows.

What this says about the design:
- **The per-call GPU sync is the cost to remove.** With a worker running next
  to training it waits for the training kernels (130–430 µs per sample);
  with no worker the GPU is idle and it costs ~4 µs — still as much as the RDMA.
- **Fixed per-call overhead dominates small rows:** the RDMA round trip is
  ~4–5 µs, while sync + allocation + Python add ~10 µs per GPU `get()`.
- **Reads are serialized** by `recv_lock`, so more threads don't add bandwidth.
- **Not worth fixing:** lock contention (≤0.2 µs) and MR registration (<1 µs
  even at 100% misses — libfabric caches registrations). No multi-entry MR cache.

A batched read pays the sync, lock, MR check and Python call once per batch
instead of once per sample, and posts all of a batch's `fi_read`s before
waiting, so the round trips overlap on the network.

Checked facts:
- torch 2.14 `_MapDatasetFetcher.fetch()` calls `dataset.__getitems__(indices)`
  if defined, so the standard DataLoader picks up a batch hook without changes.
- cxi (`fi_info -p cxi -v`, login node): tx `size: 1024`, `rma_iov_limit: 1`
  — up to 1024 outstanding ops per endpoint, one contiguous region per
  `fi_read`. Re-check on a compute node.

## 1. C++ (`include/ddstore.hpp`, `src/common.cxx`)

- New `DDStore::get_batch<T>(name, const long *idx, long n, T *buffer, int hmem_iface)`:
  row `i` of the contiguous `n`-row `buffer` receives global row `idx[i]`.
  Same per-row validation as `get()` (range, `sortedsearch` target, itemsize),
  done for all rows before anything is posted.
- method 0: loop the existing per-row `MPI_Get` path (unchanged behavior).
- methods 1/2, holding `recv_lock` once for the whole batch:
  1. Register the whole batch buffer once (`n * row_bytes`) through the existing
     recv-MR region cache.
  2. Post all `n` `fi_read`s back to back (target `comm_partner[t]`,
     `remote_address[t] + offset`, `remote_key[t]`, local `buffer + i*row_bytes`,
     same MR desc). On `-FI_EAGAIN`, drain completions with `fi_cq_read` and retry.
     Optional: coalesce runs of consecutive indices on the same target into one read.
  3. Wait for exactly `n` completions. On any error, keep draining the
     completions of reads already posted before returning — a stale completion
     left in the CQ would be consumed by the next call.
- Refactor `read_from_remote()` into `ensure_recv_mr()` + `post_read()` +
  `wait_completions(k)`; single-row `get()` = post 1 + wait 1 (same behavior).
- Extend the `DDSTORE_PROFILE` counters: count batches and rows separately so
  per-row and per-batch costs can both be reported.

## 2. Cython (`src/pyddstore.pyx`)

- `get_batch(name, arr, indices)`: `arr` shape `(n, ...)` (numpy or CUDA/HIP
  tensor), `indices` converted to a contiguous int64 array of length `n`.
- Same checks as `get()` (`_check_dtype`, contiguity, GPU fabric preconditions).
- One `torch.cuda.synchronize(device)` per batch on the GPU path (keep it
  device-wide for now: removing it entirely caused GPU faults; see Follow-ups).
- C++ call under `with nogil:`, item-size dispatch like `get()`.
- Python-side profiling as in `get()` (whole-call and sync time).

## 3. Dataset (`examples/vae/distdataset.py`, `ddstore_dataloader.py`)

- `DistDataset.__getitems__(indices)` / `DistDatasetReader.__getitems__`:
  allocate one `(n, data_disp)` buffer (numpy, or `torch.empty(..., device=...)`),
  `get_batch` data and labels, return
  `[(row_i.reshape(1, side, side), label_i) for i]` so `collate_fn` works unchanged.
- `ThreadDataLoader.fetch()`: use `dataset.__getitems__(index)` when present,
  else the per-sample list (mirror torch's fetcher).
- `DDSTORE_BATCH_GET=0` env var to fall back to per-sample `get()` for A/B.

## 4. Tests

Note: `test_single.py` / `test_multirank.py` always use `method=0`, so they
only cover `get_batch`'s MPI fallback. Add method 1 coverage explicitly.

- `test_single` / `test_multirank`: `get_batch` with shuffled indices spanning
  all ranks, duplicates, `n == 1`; compare row by row with per-row `get()`.
  Out-of-range index raises and leaves the store usable (a later `get()` works).
- Method 1 (cxi) multi-rank: the same cases, in `test_gpu_rdma.py` or a new
  libfabric test file (host destination too, not only GPU).
- `test_gpu_rdma`: same into a GPU tensor; concurrent `get_batch` from threads.

## 5. Measure (`method=1`, cxi, 2 nodes x 8 ranks, `DDSTORE_PROFILE=1`)

- `bench_get.py --batch N` (new option): rows per call 1/32/128, row sizes
  3 KB, 12.5 KB, 200 KB, 1 MB, host / GPU destination, 1–2 threads.
- `vae-ddp.py --image-scale 1` and `2` (optionally `--replicate 4`), host path and
  `--gpu-dest --gpu-source`, `--num-workers` 0/1/2, `DDSTORE_BATCH_GET` on/off.
  Check the epoch-8 loss is unchanged (8.8960 at S=1; 30.0178 at S=2).
- Baselines (per-sample `get`, epochs 2–8 avg, s/epoch):
  S=1 — host w0 0.25, host w1 0.16, GPU w0 0.30, GPU w1 0.26, GPU w2 0.40;
  S=2 — host w0 0.39, host w1 0.28, GPU w0 0.42, GPU w1 0.42, GPU w2 0.55.
- Expected: GPU path loses most of its gap (128 syncs per batch -> 1, overlapped
  reads); host path gains a little; extra workers matter even less.
- Update README (`get_batch` API, results).

## Follow-ups (after batching)

- The one remaining sync per batch still waits for training kernels when a
  worker runs. Options: a per-worker CUDA/HIP stream + stream-only sync (raise
  `GPU_MAX_HW_QUEUES`, see README), or a per-worker pre-registered receive
  ring with events. Must be re-verified against the GPU fault seen when the
  sync was removed.
- Block-shuffling sampler so consecutive rows on the same rank coalesce into
  one larger read.

## Risks / open questions

- Batch indices span many targets: 128 concurrent reads to up to 16 ranks is
  within the 1024 tx queue, but receiver-side limits are unmeasured.
- Partial-post failure paths must drain in-flight reads before the buffer is
  handed back (the subtle part).
- cxi `threading` level irrelevant here because the lock is kept.

Size estimate: C++ ~120 lines (mostly the refactor), Cython ~40, dataset ~30,
tests ~100, bench ~20.
