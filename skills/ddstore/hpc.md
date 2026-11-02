# Slurm, Frontier and Perlmutter

## Build

```bash
# Frontier (Cray MPICH; mpi4py built for it)
CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
# "No module named 'distutils.msvccompiler'": prefix SETUPTOOLS_USE_DISTUTILS=stdlib
# Without MPI (methods 1 and 2 only)
DDSTORE_MPI=0 CC=gcc CXX=g++ pip install .
```

## Job script (single srun step)

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH -t 30:00
#SBATCH --network=single_node_vni         # Frontier: needed for one-node steps, harmless otherwise

export DDSTORE_FABRIC=cxi                 # native Slingshot; required for GPUDirect
export DDSTORE_NIC_MAP=$(python3 -m cpu_nic_map --env)   # once, before srun
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)   # torch.distributed
export MASTER_PORT=29500
# export DDSTORE_FORK_WORKERS=1           # only for forked DataLoader workers

# Frontier: 8 ranks/node, 7 cores each, one GCD each
srun -N2 -n16 -c7 --gpus-per-task=1 python -u train.py
# Perlmutter: 4 ranks/node, each sees all 4 GPUs (NCCL needs it)
# srun -N2 -n8 -c32 --gpus-per-node=4 python -u train.py
```

Use `srun`, not `torchrun` or `mpirun`. Set `handshake_dir` to Lustre scratch.

## Core/extra split (method 2, two srun steps)

```bash
#SBATCH --network=single_node_vni,job_vni
WRAP='export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}; exec "$@"'   # cxi uses only the first VNI
srun -N1 -n8 ... bash -c "$WRAP" _ python core.py &
srun -N3 -n24 ... bash -c "$WRAP" _ python extra.py
```

Each step on its own nodes on Frontier. Template: `examples/vae/script/job-vae-core-extra.sh`.

## Errors

| Symptom | Cause / fix |
|---|---|
| `fi_domain()` fails with `-38`, or `no suitable fabric found` | Step has no CXI service: `#SBATCH --network=single_node_vni`. |
| `VNI_NOT_FOUND` | Reader and store in different srun steps: `job_vni` + the `SLINGSHOT_VNIS` wrapper. |
| Forked worker error about the MR cache monitor | Set `DDSTORE_FORK_WORKERS=1` before libfabric starts (job script). |
| Hang in `DistDataset(...)` | Not every rank created it, or datasets created in a different order. With method 0, unequal batch counts. |
| `variable '...' already exists in this store` / `another DistStore` | Duplicate dataset name, or method-2 file clash in `handshake_dir`. |
| Handshake timeout | `handshake_dir` not shared or slow (Perlmutter CFS); raise `DDSTORE_HANDSHAKE_TIMEOUT_S` or use scratch. |
| `DistStoreError: Unknown error 524` (Perlmutter) | torch `file://` rendezvous on CFS; use `env://` or `$SCRATCH`. |
| Slow startup in every rank | Missing `DDSTORE_NIC_MAP`. |

## Useful variables

| Variable | Purpose |
|---|---|
| `DDSTORE_METHOD` | Default method. |
| `DDSTORE_COMM` | `mpi`, `torch` or `file`. |
| `DDSTORE_FABRIC` | `hsn` (TCP, default) or `cxi`. |
| `DDSTORE_HANDSHAKE_DIR` | Default `handshake_dir`. |
| `DDSTORE_PROFILE=1` | Read timing counters (`ds.get_profile(name)`). |
| `DDSTORE_BATCH_GET=0` | Read per sample instead of per batch (debugging). |
| `GPU_MAX_HW_QUEUES` | ROCm: raise if loader threads use their own streams. |
