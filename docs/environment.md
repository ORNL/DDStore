# Environment variables

**At build time:** `DDSTORE_MPI=0` builds without MPI (methods 1 and 2 only; see [Building without MPI](installation.md#building-without-mpi)). Default `1`.

**Read by DDStore itself** (the C++ library, `pyddstore`, `cpu_nic_map`):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_FABRIC` | `hsn` | libfabric provider for `method=1`/`2`: `hsn` (`tcp;ofi_rxm`) or `cxi` (native Slingshot; required for [GPUDirect RDMA](gpudirect.md)). See [libfabric RDMA](backends.md#libfabric-rdma-method1). |
| `FABRIC_IFACE` | auto | Network interface (libfabric domain, e.g. `cxi0`, `hsn0`) for `method=1`/`2`. Set it to force one; otherwise picked from the rank's CPU affinity. |
| `DDSTORE_NIC_MAP` | unset | Precomputed CPU→NIC map used for that automatic pick instead of a live hwloc query (`python3 -m cpu_nic_map --env`). The constructor's `nic_map=` argument takes priority. |
| `DDSTORE_HANDSHAKE_DIR` | `./ddstore_hs` | `method=2` handshake directory when none is given (C++ API; `PyDDStore` requires `handshake_dir`, and the examples fill it from this variable). Must be on a shared filesystem. |
| `DDSTORE_HANDSHAKE_TIMEOUT_S` | `300` | Seconds a `method=2` extra member's `join()` polls for the core group's record file, and a rank [without MPI](backends.md#without-mpi-methods-1-and-2) waits in a file exchange. |
| `DDSTORE_SESSION` | `{SLURM_JOB_ID}_{SLURM_STEP_ID}`, else `0` | Without MPI: part of the exchange file names, so runs sharing a directory don't collide. Must be equal on all ranks of a store. |
| `DDSTORE_PROFILE` | off | `1` turns on `get()`/`get_batch()` timing counters, read with `get_profile(name)`. See [Performance](performance.md). |
| `DDSTORE_MAX_READ_BYTES` | `1073741824` (1 GiB) | `method=1`/`2`: largest single `fi_read`; longer rows are read in pieces (on Perlmutter's `cxi` one 5 GB read fails with `EMSGSIZE`, 2.5 GB works, and the provider doesn't report the limit). Lowered to the endpoint's `max_msg_size` when the provider reports one. |
| `DDSTORE_ALLTOALL_MAX_BYTES` | `2097152` (2 MiB) | `method=0` `get_batch()`: bytes each rank receives per exchange round. Must be equal on all ranks. |
| `DDSTORE_FORK_WORKERS` | unset | `1`: stores will be read from forked `DataLoader` workers. Sets `FI_MR_CACHE_MONITOR=memhooks` if it is unset or `kdreg2` (Frontier's default, which hangs forked workers). Must be set before libfabric starts. See [Forked DataLoader workers](pytorch.md#forked-dataloader-workers). |
| `FI_CXI_DISABLE_HOST_REGISTER` | set to `1` by `pyddstore` and DDStore unless already set | libfabric CXI: don't register host receive buffers with ROCm. Needed for forked workers (registration fails after fork). |

**Read by `pyddstore.torch`** (defaults for arguments not given):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_METHOD` | `0` with MPI, `1` without | `DistDataset`'s backend when `method=` isn't passed: `0` MPI RMA, `1` libfabric, `2` file-based handshake. (`PyDDStore` itself takes `method=` only.) |
| `DDSTORE_COMM` | `mpi` in an MPI build, else `torch` if `torch.distributed` is initialized, else `file` | `DistDataset`'s communicator when `comm=None`: `mpi` (`MPI.COMM_WORLD`), `torch` (the default `torch.distributed` group) or `file` (a `FileComm` in `handshake_dir`, rank and size from `SLURM_PROCID`/`SLURM_NTASKS`, else `RANK`/`WORLD_SIZE` or PMI). The examples' core server reads it too. |
| `DDSTORE_BATCH_GET` | `1` | `DistDataset.__getitems__` reads a whole batch with one `get_batch()` per field; `0` reads one sample at a time. |
| `DDSTORE_HANDSHAKE_DIR`, `DDSTORE_HANDSHAKE_TIMEOUT_S` | `./ddstore_hs`, `300` | `method=2` directory (also the exchange directory without MPI), and how long `DistDatasetReader` waits for the core group to publish. |
| `DDSTORE_N_CORE` | unset | `DistDatasetReader`'s number of core ranks when `n_core=` isn't passed (the examples default it to 4). |
| `DDSTORE_THREAD_READERS` | `1` | `ThreadDataLoader`, methods 1/2: each worker thread reads through its own reader (`fork_reader()`, own endpoint) instead of sharing the store and its per-variable lock. `0` = share. |
| `DDSTORE_AFFINITY_WIDTH` / `DDSTORE_AFFINITY_OFFSET` | `0` / `0` | `ThreadDataLoader`: pin worker thread *i* to CPUs `[offset + i·width, offset + (i+1)·width)` of the process's affinity; width `0` = no pinning. |

**Read by the examples** (`examples/vae/`, `examples/scripts/`, job scripts):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_BACKEND` | auto | `torch.distributed` backend for the examples' DDP setup (`nccl`, `gloo`, `xccl`). |
| `VAE_PROFILE` | off | `1`: `vae-ddp.py` prints per-epoch fetch vs compute time. |
| `MASTER_PORT` | `2345` | DDP rendezvous port; the core/extra job script gives each step its own. |

**System settings that matter on Frontier:**

| Variable | Effect |
|---|---|
| `SLINGSHOT_VNIS` | Set by Slurm per step. With `--network=job_vni`, keep only the last (job-wide) entry before starting Python so separate `srun` steps can reach each other — see [Multiple `srun` steps](hpc.md#multiple-srun-steps-in-one-job-method2-cxi). |
| `GPU_MAX_HW_QUEUES` | ROCm hardware queues per GPU per process (default 4); raise it if data-loading threads use their own streams — see [HIP streams](results.md#hip-streams-and-hardware-queues-frontier-rocm-72). |
