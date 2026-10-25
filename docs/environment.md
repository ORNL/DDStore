# Environment variables

**At build time:** `DDSTORE_MPI=0` builds without MPI (methods 1 and 2 only; see [Building without MPI](installation.md#building-without-mpi)). Default `1`.

**Read by DDStore itself** (the C++ library, `pyddstore`, `cpu_nic_map`):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_FABRIC` | `hsn` | libfabric provider for `method=1`/`2`: `hsn` (`tcp;ofi_rxm`) or `cxi` (native Slingshot; required for [GPUDirect RDMA](gpudirect.md)). See [libfabric RDMA](backends.md#libfabric-rdma-method1). |
| `FABRIC_IFACE` | auto | Network interface for `method=1`/`2` (e.g. `cxi0`, `hsn0`); default from the rank's CPU affinity. |
| `DDSTORE_NIC_MAP` | unset | Precomputed CPU→NIC map (`python3 -m cpu_nic_map --env`) instead of a live hwloc query; `nic_map=` takes priority. |
| `DDSTORE_HANDSHAKE_DIR` | `./ddstore_hs` | `method=2` handshake directory (C++ API and examples; `PyDDStore` takes `handshake_dir`). Shared filesystem. |
| `DDSTORE_HANDSHAKE_TIMEOUT_S` | `300` | Seconds `join()` and file exchanges [without MPI](backends.md#without-mpi-methods-1-and-2) wait. |
| `DDSTORE_SESSION` | `{SLURM_JOB_ID}_{SLURM_STEP_ID}`, else `0` | Without MPI: exchange file prefix, so runs can share a directory. Same on all ranks. |
| `DDSTORE_PROFILE` | off | `1`: `get()`/`get_batch()` timing counters (`get_profile(name)`). See [Performance](performance.md). |
| `DDSTORE_MAX_READ_BYTES` | `1073741824` (1 GiB) | `method=1`/`2`: largest single `fi_read`; longer rows are split (Perlmutter `cxi` rejects 5 GB reads without reporting a limit). Capped at the provider's `max_msg_size`. |
| `DDSTORE_ALLTOALL_MAX_BYTES` | `2097152` (2 MiB) | `method=0` `get_batch()`: bytes per rank per exchange round. Same on all ranks. |
| `DDSTORE_FORK_WORKERS` | unset | `1`: allow reads from forked `DataLoader` workers; sets `FI_MR_CACHE_MONITOR=memhooks` if unset or `kdreg2`. Set before libfabric starts. See [Forked DataLoader workers](pytorch.md#forked-dataloader-workers). |
| `FI_CXI_DISABLE_HOST_REGISTER` | `1` (set by `pyddstore` unless already set) | CXI: skip ROCm registration of host buffers; needed for forked workers. |

**Read by `pyddstore.torch`** (defaults for arguments not given):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_METHOD` | `0` with MPI, `1` without | `DistDataset`'s default `method` (`PyDDStore` takes `method=` only). |
| `DDSTORE_COMM` | `mpi` in an MPI build, else `torch` if initialized, else `file` | `DistDataset`'s communicator for `comm=None`: `MPI.COMM_WORLD`, the default `torch.distributed` group, or a `FileComm` (rank/size from Slurm, `RANK`/`WORLD_SIZE` or PMI). |
| `DDSTORE_BATCH_GET` | `1` | `__getitems__` reads a batch with one `get_batch()` per field; `0`: per sample. |
| `DDSTORE_HANDSHAKE_DIR`, `DDSTORE_HANDSHAKE_TIMEOUT_S` | `./ddstore_hs`, `300` | Default `handshake_dir`, and `DistDatasetReader`'s wait for the core group. |
| `DDSTORE_N_CORE` | unset | `DistDatasetReader`'s default `n_core` (examples: 4). |
| `DDSTORE_THREAD_READERS` | `1` | `ThreadDataLoader`, methods 1/2: each worker thread reads through its own reader, so threads read in parallel. `0`: threads share the store. |
| `DDSTORE_FAST_COLLATE` | `1` | `ThreadDataLoader` with the default `collate_fn`: read a batch directly into its batch tensors. `0`: build per-sample values and collate them. |
| `DDSTORE_AFFINITY_WIDTH` / `DDSTORE_AFFINITY_OFFSET` | `0` / `0` | `ThreadDataLoader`: pin worker *i* to CPUs `[offset + i·width, offset + (i+1)·width)`; `0` = off. |

**Read by the examples** (`examples/vae/`, `examples/scripts/`, job scripts):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_BACKEND` | auto | Examples' DDP backend (`nccl`, `gloo`, `xccl`). |
| `VAE_PROFILE` | off | `1`: `vae-ddp.py` prints fetch and compute time per epoch. |
| `MASTER_PORT` | `2345` | DDP rendezvous port (one per step in the core/extra script). |

**System settings that matter on Frontier:**

| Variable | Effect |
|---|---|
| `SLINGSHOT_VNIS` | Set by Slurm per step. With `--network=job_vni`, keep only the last (job) entry so separate steps can connect — [Slingshot VNIs](hpc.md#slingshot-vnis-and---network). |
| `GPU_MAX_HW_QUEUES` | ROCm hardware queues per GPU per process (default 4); raise it if loader threads use their own streams — [HIP streams](results.md#hip-streams-and-hardware-queues-frontier-rocm-72). |
