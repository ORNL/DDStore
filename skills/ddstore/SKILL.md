---
name: ddstore
description: Add DDStore (pyddstore) to a PyTorch training code so each rank loads only its share of the dataset and reads the rest over RDMA. Use when integrating DistDataset/DistStore into a training loop, choosing a DDStore method (0/1/2), communicator (MPI, torch.distributed, files) or loader (ThreadDataLoader, forked DataLoader workers), writing Slurm job scripts for it (Frontier, Perlmutter), or debugging DDStore build, fabric or handshake errors.
---

# DDStore integration

DDStore keeps a dataset in the aggregate memory of the ranks: each rank loads 1/N of the samples once, at startup, and any rank reads any sample by global index (MPI RMA or libfabric RDMA). `pyddstore.torch.DistDataset` wraps a map-style dataset. The dataset, sampler, loader and launch change; the model and training math don't.

Files in this skill's directory (paths below are relative to it):
- [api.md](api.md): signatures and arguments.
- [patterns.md](patterns.md): code for the common setups.
- [hpc.md](hpc.md): Slurm, Frontier and Perlmutter settings, errors.
- `scripts/check_env.py`: build, methods, resolved communicator and settings. Run it first: `python <this skill's directory>/scripts/check_env.py`.

Full docs: https://github.com/ORNL/DDStore/tree/main/docs. Complete examples: [examples/minimal/train.py](https://github.com/ORNL/DDStore/blob/main/examples/minimal/train.py) (about 100 lines; start here), [examples/vae/vae-ddp.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae-ddp.py) (all options).

## Steps

1. **Check the environment.** Run `scripts/check_env.py` (in a job step if possible). Note `HAVE_MPI`, the communicator `comm=None` resolves to, and the default method.

2. **Find the dataset, its transforms and the loader.** Locate the map-style `Dataset` (`len()` + `[i]`) and the `DataLoader`/sampler. For an `IterableDataset`, ask the user how to index it.
   - Each field must keep the same shape and dtype across samples (bool, uint8, int32, int64, float32, float64, numpy records). For strings, labels or variable metadata, use `encode`/`decode`.
   - **Split the transforms.** `DistDataset` calls `source[i]` once per sample at construction and stores the result. Deterministic work (file decode, fixed-shape resize) belongs in the source. Random or epoch-dependent work (crops, flips, augmentation, masking) must move after the read ([patterns.md](patterns.md#4-random-augmentation)); otherwise every epoch sees the same augmented sample, with no error. Keep the model's inputs identical: same dtype, scale, normalization and shape as the original pipeline. If the original resize doesn't give a fixed shape (e.g. `Resize(256)` keeps the aspect ratio), ask the user which fixed-shape option to use.

3. **Pick the method.**

   | Method | Needs | Use when |
   |---|---|---|
   | 0 | MPI communicator | Reads are collective: every rank iterates the same number of batches (`DistributedSampler` does), from exactly one thread. |
   | 1 | libfabric | Independent reads; works with forked `DataLoader` workers. GPU destinations need `DDSTORE_FABRIC=cxi`. |
   | 2 | libfabric + shared `handshake_dir` | Same as 1, plus a separate "extra" job can read the data (`DistDatasetReader`). |

   Default: `DDSTORE_METHOD`, else 0 with an MPI communicator and 1 otherwise. Without a reason to choose otherwise, use method 1.

4. **Pick the communicator.** Pass what the code already has:
   - mpi4py: `MPI.COMM_WORLD` (MPI build).
   - DDP: `dist.group.WORLD` (set the CUDA device first with NCCL/RCCL). Any backend works.
   - `None`: `MPI.COMM_WORLD` in an MPI build; otherwise the default torch group if initialized, else files in `handshake_dir`. `DDSTORE_COMM=mpi|torch|file` overrides.

5. **Wrap the datasets.** One `DistStore` for all datasets of a run; distinct names; create them in the same order on every rank; `store.free()` at the end. See [patterns.md](patterns.md).

6. **Pick the loader.**
   - `ThreadDataLoader` (default choice): worker threads, reads straight into GPU memory with `device=`. Methods 1/2: 1–2 workers are usually enough. Method 0: `num_workers=1` exactly (0 also means one thread). That thread calls MPI, so method 0 requires a provided MPI thread level of `SERIALIZED` or higher (check with `MPI.Query_thread()`; the requested level or mpi4py's default `MULTIPLE` doesn't prove what MPI provides). With `SERIALIZED`, no other thread may call MPI while the loader reads, on any communicator. With `MULTIPLE`, other threads may call MPI meanwhile, on any communicator: the loader's exchanges run on a private duplicate of the store's communicator, made by `add()`, and make no collective on the communicator passed in.
   - `torch.utils.data.DataLoader` with `num_workers=0`: reads on the calling thread; any method.
   - `torch.utils.data.DataLoader` with forked workers: methods 1/2 only; set `DDSTORE_FORK_WORKERS=1` before libfabric starts; host memory only (`device=` ignored); use `persistent_workers=True`.
   - Never `spawn` or `forkserver` start methods (they don't work with MPI on Frontier).

7. **Launch.** On Slurm systems use `srun`, not `torchrun` or `mpirun`; see [hpc.md](hpc.md) for flags and `DDSTORE_NIC_MAP`. Elsewhere use `mpirun` (MPI build) or a launcher that sets `RANK`/`WORLD_SIZE`.

8. **Verify.** Compare the loss of the first epochs with the original code at the same seed and rank count. Time the fetch separately (`DDSTORE_PROFILE=1` and `ds.get_profile(name)`, or time the loader loop).

## Rules

- DDStore stores what `source[i]` returns, once. Random transforms in the source are frozen: move them after the read, unless the user wants fixed, precomputed augmentation.
- Method 0: one loader thread at most.
- Import order: `torch`, then `pyddstore`, then `mpi4py`.
- `DistStore(...)`, `DistDataset(...)` and `store.free()` are collective: every rank of the communicator calls them, in the same order.
- Don't pass `comm`, `method`, `handshake_dir` or `ddstore_width` to a `DistDataset` together with a `DistStore`; they belong to the store.
- Method 2: dataset names map to files `{name}.meta.json` and `{name}_{field}.bin` (`.` and `/` become `_`); two live datasets can't publish the same files in one `handshake_dir`.
- `handshake_dir` must be on a shared filesystem (Lustre scratch; not Perlmutter CFS).
- Don't change the user's model or training math.
