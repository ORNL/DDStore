---
name: ddstore
description: Add DDStore (pyddstore) to a PyTorch training code so each rank loads only its share of the dataset and reads the rest over RDMA. Use when integrating DistDataset/DistStore into a training loop, choosing a DDStore method (0/1/2), communicator (MPI, torch.distributed, files) or loader (ThreadDataLoader, forked DataLoader workers), writing Slurm job scripts for it (Frontier, Perlmutter), or debugging DDStore build, fabric or handshake errors.
---

# DDStore integration

DDStore keeps a dataset in the aggregate memory of the ranks: each rank loads 1/N of the samples, and any rank reads any sample by global index (MPI RMA or libfabric RDMA). `pyddstore.torch.DistDataset` wraps a map-style dataset; training code otherwise stays the same.

References in this folder:
- [api.md](api.md): signatures and arguments.
- [patterns.md](patterns.md): code for the common setups.
- [hpc.md](hpc.md): Slurm, Frontier and Perlmutter settings, errors.
- `scripts/check_env.py`: reports the build, methods available and relevant settings. Run it first.

Full docs: https://ornl.github.io/DDStore (source in `docs/` of https://github.com/ORNL/DDStore).

## Steps

1. **Check the environment.** `python scripts/check_env.py`. Note `HAVE_MPI` (an MPI build or `DDSTORE_MPI=0`) and whether `torch` imports.

2. **Find the dataset and loader.** Locate the map-style `Dataset` (`len()` + `[i]`) and the `DataLoader`/sampler. DDStore needs map-style; for an `IterableDataset`, ask the user how to index it. Each field must keep the same shape and dtype across samples (bool, uint8, int32, int64, float32, float64, numpy records). For strings, labels or variable metadata, use `encode`/`decode`.

3. **Pick the method.**

   | Method | Needs | Use when |
   |---|---|---|
   | 0 | MPI | MPI build, `ThreadDataLoader` or no workers. Reads are collective: every rank must iterate the same number of batches (`DistributedSampler` does). |
   | 1 | libfabric | Default without MPI. Independent reads; works with forked `DataLoader` workers. |
   | 2 | libfabric + shared `handshake_dir` | Same as 1, plus a separate "extra" job can read the data (`DistDatasetReader`). |

   Without a reason to choose otherwise, use method 1.

4. **Pick the communicator.** Pass what the code already has:
   - mpi4py: `MPI.COMM_WORLD` (MPI build).
   - DDP: `dist.group.WORLD` (call `torch.cuda.set_device()` first with NCCL/RCCL). Any backend works.
   - Neither: `None` (files in `handshake_dir`, ranks from Slurm).

5. **Wrap the datasets.** One `DistStore` for all datasets of a run; distinct names; create them in the same order on every rank; `store.free()` at the end. See [patterns.md](patterns.md).

6. **Pick the loader.**
   - `ThreadDataLoader` (default choice): worker threads, reads straight into GPU memory with `device=`, works with all methods. 1–2 workers are usually enough.
   - `torch.utils.data.DataLoader` with forked workers: methods 1/2 only; set `DDSTORE_FORK_WORKERS=1` before libfabric starts; host memory only (`device=` ignored); use `persistent_workers=True`.
   - Never `spawn` or `forkserver` start methods (they don't work with MPI on Frontier).

7. **Launch with `srun`**, not `torchrun`. See [hpc.md](hpc.md) for flags and `DDSTORE_NIC_MAP`.

8. **Verify.** Compare the loss of the first epochs with the original code at the same seed and rank count. Time the fetch separately (`DDSTORE_PROFILE=1` and `ds.get_profile(name)`, or time the loader loop).

## Rules

- Import order: `torch`, then `pyddstore`, then `mpi4py`.
- `DistDataset(...)`, `DistStore(...)` and `free()` are collective: every rank of the communicator calls them.
- Don't pass `comm`, `method`, `handshake_dir` or `ddstore_width` to a `DistDataset` together with a `DistStore`; they belong to the store.
- Method 2: dataset names map to files `{name}.meta.json` and `{name}_{field}.bin` (`.` and `/` become `_`); two live datasets can't publish the same files in one `handshake_dir`.
- `handshake_dir` must be on a shared filesystem (Lustre scratch; not Perlmutter CFS).
- Don't change the user's model or training math. Change only the dataset, sampler, loader and launch.
