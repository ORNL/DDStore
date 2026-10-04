# DDStore

<img src="images/DDStore-logo.png" alt="DDStore logo" />

Efficient distributed data loading for distributed data-parallel (DDP) training.

Each MPI rank holds a shard of the full dataset in memory. DDStore exposes a global index space so any rank can read any sample via one-sided remote memory access — either MPI RMA (default) or libfabric RDMA — without coordinator synchronization.

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="DDStore architecture" width="300" />

## Prerequisites

| Dependency | Notes |
|---|---|
| MPI (OpenMPI / MPICH) | `mpicc` and `mpicxx` must be on `PATH` |
| libfabric | Required for the RDMA backends (`method=1` and `method=2`) |
| Python ≥ 3.6 | |
| NumPy, mpi4py, Cython | Python build dependencies |

## Installation

```bash
# Install Python build dependencies
pip install numpy mpi4py Cython

# Build in-place (use with PYTHONPATH=$PWD:$PYTHONPATH)
CC=mpicc CXX=mpicxx python setup.py build_ext --inplace

# Or install into the active virtual environment
CC=mpicc CXX=mpicxx pip install .

# Or install in editable/development mode
CC=mpicc CXX=mpicxx pip install -e .

# Or install directly from GitHub
CC=mpicc CXX=mpicxx pip install git+https://github.com/ORNL/DDStore.git
```

To build against the packages already in the current environment (e.g. an `mpi4py` built against Cray MPICH) instead of letting pip fetch fresh build dependencies into an isolated build environment, disable build isolation:

```bash
CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
```

If that fails with `ModuleNotFoundError: No module named 'distutils.msvccompiler'` (newer setuptools combined with an older system NumPy, e.g. `cray-python/3.11.7` on Frontier), point setuptools at the standard-library `distutils` for the build:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
```

## Quick Start

```python
import numpy as np
from mpi4py import MPI
import pyddstore as dds

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each rank contributes its own shard
store = dds.PyDDStore(comm)                  # MPI RMA backend (default)
# store = dds.PyDDStore(comm, method=1)      # libfabric RDMA backend

data = np.random.rand(1024, 64).astype(np.float32)
store.add("features", data)                  # collective — all ranks must call

# Read any global sample index
out = np.zeros((1, 64), dtype=np.float32)
store.epoch_begin()
store.get("features", out, start=2048)       # global index across all shards
store.epoch_end()

store.free()
```

Run with:
```bash
mpirun -n 4 python my_script.py
```

## API Reference

### `PyDDStore(comm_or_none=None, method=0, handshake_dir="", n_core=0, nic_map=None)`

| Parameter | Type | Description |
|---|---|---|
| `comm_or_none` | `mpi4py.MPI.Comm` or `None` | MPI communicator covering all ranks. `None` only for a `method=2` extra member |
| `method` | `int` | `0` = MPI RMA (default), `1` = libfabric RDMA, `2` = file-based handshake (see [below](#file-based-handshake-method2)) |
| `handshake_dir` | `str` | Required for `method=2`: shared-filesystem directory used to exchange RDMA addresses |
| `n_core` | `int` | Required for a `method=2` extra member: number of core ranks that published data |
| `nic_map` | `str` or `None` | Optional, `method=1`/`2` only: a precomputed CPU→NIC map string (see [`DDSTORE_NIC_MAP`](#libfabric-rdma-method1) below) to use instead of the environment variable. Ignored if `FABRIC_IFACE` is already set |

Four call shapes:

```python
PyDDStore(comm)                                     # method 0, MPI RMA
PyDDStore(comm, method=1)                            # method 1, libfabric RDMA
PyDDStore(comm, method=2, handshake_dir="/path")      # method 2, core member (n_core == comm size)
PyDDStore(None, method=2, handshake_dir="/path", n_core=N)  # method 2, extra member (no comm)
```

Note: grouping ranks into independent stores (the "sub-communicator" pattern below) is done by splitting `comm` yourself before constructing `PyDDStore` — there is no `ddstore_width` constructor parameter. `DistDataset` in [examples/vae/distdataset.py](examples/vae/distdataset.py) shows the pattern (`comm.Split()` then `PyDDStore(sub_comm)`).

---

### `init(name, nrows, disp, itemsize=1)`

Pre-allocate a named variable without providing data yet. Use `update()` to fill it in afterwards. **Collective**.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `nrows` | `int` | Number of rows in this rank's shard |
| `disp` | `int` | Number of elements per row |
| `itemsize` | `int` | Bytes per element (default `1`) |

---

### `add(name, arr)`

Register a NumPy array as a named variable. Each rank contributes its local shard; the global index space is the concatenation of all shards in rank order. **Collective** — all ranks in `comm` must call with the same `name`.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` or `torch.Tensor` | C-contiguous 2-D (or 1-D) array/tensor. Supported dtypes: `int32`, `int64`, `uint8`, `float32`, `float64`, `bool_`/`bool`. A CUDA/HIP tensor registers GPU memory directly — see [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers) below |

---

### `update(name, arr, offset)`

Overwrite a region of the local shard for a variable registered with `init()`. Local operation — does not require epoch or barrier.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` | Data to write |
| `offset` | `int` | Row offset within the local shard |

---

### `get(name, arr, start=0)`

Read `arr.shape[0]` consecutive rows starting at global index `start` into `arr`. The range must fall within a single rank's shard. Must be called inside an `epoch_begin` / `epoch_end` pair when using the MPI backend.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` or `torch.Tensor` | Pre-allocated, C-contiguous output buffer. A CUDA/HIP tensor writes the RDMA transfer directly into GPU memory — see [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers) below |
| `start` | `int` | Global row index |

---

### `get_batch(name, arr, indices)`

Read rows `indices` (global row ids; any order, any ranks, repeats allowed) into `arr`: row `i` of `arr` receives row `indices[i]`, so `arr.shape[0]` must equal `len(indices)` (else `ValueError`). Same buffer rules as `get()` (NumPy array or CUDA/HIP tensor). Every index is checked before anything is read; an out-of-range one raises `IndexError` and leaves the store usable.

For `method=1`/`2` the whole batch is one call: one lock acquisition, one memory registration, and (GPU destination) one device sync, with all of the batch's `fi_read`s posted before any is waited for, so the reads overlap on the network. It is still one `fi_read` per row. `method=0` loops the per-row `MPI_Get` path.

```python
idx = np.array([2048, 7, 4096, 7])
out = np.zeros((len(idx), 64), dtype=np.float32)
store.get_batch("features", out, idx)
```

`DistDataset`/`DistDatasetReader` use it by default through `__getitems__`, which PyTorch's `DataLoader` (and `ThreadDataLoader`) calls with a whole batch's indices, so the VAE examples and job scripts read in batches with no extra flag. Set `DDSTORE_BATCH_GET=0` to fall back to one `get()` per sample.

---

### `join(name)`

`method=2` extra member only. Discovers a variable published by the core group by polling the handshake directory until the combined record file (`{name}.bin`) written by core rank 0 reaches its expected size (up to `DDSTORE_HANDSHAKE_TIMEOUT_S` seconds), then registers it for `get()`.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier, matching the `name` used in the core group's `add()` |

---

### `info(name)`

Returns `(total_rows, disp, itemsize)` for a variable that has been `add()`-ed or `join()`-ed. Useful on the extra side to size output buffers without hardcoding shapes.

---

### `epoch_begin()` / `epoch_end()`

Open and close an MPI RMA access epoch (calls `MPI_Win_fence`). **Collective**. Required around `get()` calls when using `method=0`. No-op for `method=1`/`2`.

---

### `free()`

Release every variable's MPI window (`method=0`) or libfabric endpoints and memory registrations (`method=1`/`2`), then the host buffer DDStore allocated for it in `add()`/`init()` (a GPU tensor passed to `add()` is the caller's and is not freed). Safe to call more than once. After `MPI_Finalize` the MPI window and buffer can no longer be released and are skipped.

## Backends

### MPI RMA (`method=0`, default)

Uses `MPI_Win_create` and `MPI_Get` for one-sided remote reads. Works on any MPI-capable cluster without additional hardware. `epoch_begin`/`epoch_end` are required to delimit access epochs.

### libfabric RDMA (`method=1`)

Uses `fi_read` for true RDMA transfers over high-speed interconnects (Infiniband/verbs, Cray GNI, Intel PSM2, Cray Slingshot). Lower latency than MPI RMA on supported hardware. `epoch_begin`/`epoch_end` are no-ops with this backend.

**`DDSTORE_FABRIC`** selects which libfabric provider to open, for `method=1`/`2`:

- `hsn` (default, unset) — opens the `tcp;ofi_rxm` domain over Cray Slingshot (Frontier).
- `cxi` — opens the native `cxi` domain over Cray Slingshot (Frontier and Perlmutter; Perlmutter is CXI-only). Required for [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers), and the default in the Frontier job scripts.

The two are independent code paths (not runtime auto-detection), so set this explicitly per system rather than relying on a guess:

```bash
export DDSTORE_FABRIC=hsn   # tcp;ofi_rxm (default)
export DDSTORE_FABRIC=cxi   # native CXI: Perlmutter, or Frontier with GPUDirect
```

`PyDDStore` picks the network interface (`FABRIC_IFACE`) automatically for `method=1`/`2`, based on each rank's real CPU affinity (`os.sched_getaffinity`) — no changes needed in your code:

- **`DDSTORE_NIC_MAP`** — a precomputed CPU→NIC map, used directly if set (no NIC discovery at construction time). Generate it once from a context with reliable NIC visibility, e.g. an `sbatch` batch step's own shell (not a nested `srun` task — NIC/PCI discovery has been observed to fail there), and export it before launching ranks so every one inherits it:
  ```bash
  export DDSTORE_NIC_MAP=$(python3 -m cpu_nic_map --env)
  srun ... python train.py
  ```
- If `DDSTORE_NIC_MAP` isn't set, each rank falls back to a live `hwloc-calc`/`lstopo` query against its own CPU affinity (`cpu_nic_map.allocated_nics()`, also runnable standalone as `python3 cpu_nic_map.py --allocated`) to find the nearest NIC.
- Set `FABRIC_IFACE` explicitly to override both and force a specific interface, e.g. when the automatic selection picks the wrong one:
  ```bash
  export FABRIC_IFACE=hsn0   # e.g. Cray Slingshot
  ```
- Or skip the environment entirely and pass a map straight to the constructor: `PyDDStore(comm, method=1, nic_map="hsn0=0-15,64-79;hsn1=...")`.

### File-based handshake (`method=2`)

Splits the dataset-holding job from the training job entirely: a **core** group loads and publishes data, and a separate **extra** group reads it over RDMA (`fi_read`, same transport as `method=1`) — the two are independent MPI jobs (e.g. two separate `srun`/`mpirun` launches, possibly on different node allocations) that never share a communicator. They rendezvous only through record files written to a shared-filesystem directory (must be visible to all nodes, e.g. Lustre):

- **Core member** — has an MPI communicator, publishes with `add()`/`init()`. Core ranks exchange records via `MPI_Allgather`, and rank 0 writes the combined set to a single `{name}.bin` file (fabric address, MR key, base pointer, row count, dtype per rank) into `handshake_dir`.
- **Extra member** — no MPI communicator; constructed with `comm_or_none=None` and an explicit `n_core`. Calls `join(name)` to poll for and read all `n_core` core-rank records, then `get()` works exactly as on the core side, reading directly from core-rank memory over RDMA.

```python
# core side — one MPI job
store = dds.PyDDStore(comm, method=2, handshake_dir="/lustre/.../ddstore_hs")
store.add("x", data)
...                       # wait for the extra side to finish (e.g. a sentinel file)
store.free()

# extra side — a separate MPI job, no comm needed
store = dds.PyDDStore(None, method=2, handshake_dir="/lustre/.../ddstore_hs", n_core=4)
store.join("x")
out = np.zeros((1, ncols), dtype=np.float32)
store.get("x", out, start=global_idx)
store.free()
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `DDSTORE_HANDSHAKE_DIR` | `./ddstore_hs` | Shared directory for handshake record files |
| `DDSTORE_HANDSHAKE_TIMEOUT_S` | `300` | Seconds to poll for core records / a join before raising a timeout |
| `DDSTORE_NIC_MAP` | unset | CPU→NIC map for `FABRIC_IFACE` auto-selection — see [libfabric RDMA](#libfabric-rdma-method1) above |
| `DDSTORE_FABRIC` | `hsn` | `hsn` (`tcp;ofi_rxm`) or `cxi` (native CXI) — see [libfabric RDMA](#libfabric-rdma-method1) above |

See [test/test_method2_core.py](test/test_method2_core.py) / [test/test_method2_extra.py](test/test_method2_extra.py) for a minimal runnable pair, and [examples/vae/vae_core_server.py](examples/vae/vae_core_server.py) / [examples/vae/vae_extra_train.py](examples/vae/vae_extra_train.py) for a full DDP training example using this split.

`ddstore_width` grouping (below) is not currently supported with `method=2` — every core rank in `comm` is treated as one group.

## GPUDirect RDMA (GPU-resident buffers)

`add()` and `get()` accept a CUDA/HIP `torch.Tensor` in place of a NumPy array, letting RDMA read from or write directly into GPU memory — no host staging buffer, no `.cpu()`/`.to(device)` copy. Requires `method=1` or `2` and a CUDA- or ROCm/HIP-enabled PyTorch build.

**Requires `DDSTORE_FABRIC=cxi`. The `hsn` provider does not support GPUDirect RDMA at all** — passing a GPU tensor to `add()`/`get()` while `DDSTORE_FABRIC=hsn` (the default) raises a clear error rather than silently falling back to a host copy.

```python
import torch
data = torch.rand(1024, 64, dtype=torch.float32, device="cuda")
store.add("features", data)                      # GPU source -- no host copy

out = torch.empty((1, 64), dtype=torch.float32, device="cuda")
store.get("features", out, start=2048)            # GPU destination -- no host copy
```

Passing a GPU tensor to `add()` registers a **raw pointer into your tensor's own storage — no copy is made.** You must keep that tensor alive (not garbage-collected, not reused) for as long as the variable stays registered, i.e. until `free()`. `PyDDStore` holds its own reference internally as a safety net, but calling `add()` again for the same variable name with a GPU tensor is rejected outright rather than silently dropping the earlier reference. This differs from the NumPy path, where `add()` always makes a private copy and the caller's array can be freed or reused immediately after the call returns. `get()`'s destination buffer has no such caveat — it's yours as usual.

Not supported: `init()`/`update()` (the incremental-fill path) remain host-only; `method=0` (MPI RMA) does not support GPU buffers on either `add()` or `get()`. Both raise a clear error naming the actual requirement if you try.

See [test/test_gpu_rdma.py](test/test_gpu_rdma.py) for runnable examples covering both directions, both libfabric methods, and the negative/error cases, and the `--gpu-dest`/`--gpu-source` flags on [examples/vae/vae-ddp.py](examples/vae/vae-ddp.py) / [examples/vae/vae_extra_train.py](examples/vae/vae_extra_train.py) / [examples/vae/vae_core_server.py](examples/vae/vae_core_server.py) for a full DDP training example using it.

GPU kernels execute asynchronously: a compute kernel that just wrote to (or is about to read) a buffer may not have fully retired by the time that buffer is handed to RDMA. On at least one ROCm+CXI build, this produced a real, confirmed bug: the RDMA transfer reported success, but the destination buffer could still show stale, pre-transfer content, because the GPU's cache hadn't been reconciled with the external NIC write. Under sustained, real-workload conditions (not just short unit tests) this showed up as hard GPU faults, not just wrong data. To guard against this, `PyDDStore` always synchronizes the GPU device (`torch.cuda.synchronize()`) before registering a buffer for RDMA in `add()`/`get()`. This is a blocking, whole-device sync, which can serialize GPU compute against RDMA transfers when called at high frequency (e.g. once per sample in a data loader) — see the performance note below.

The sync in `get()` was re-checked by removing it: every `--gpu-dest` run of `vae-ddp.py` (`method=1`, `cxi`, 2 Frontier nodes, any `--num-workers` including 0) aborted during the first epoch with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` (GPU memory fault) on every rank, while host-path and `--gpu-source`-only runs were unaffected. With the sync restored the same runs complete normally. Keep it.

## Known Limitations

### Multiple `srun` steps in one job (`method=2`, `cxi`, Frontier)

Core and extra run as separate `srun` steps, and on Slingshot every step gets its own VNI (network isolation ID); two endpoints can only communicate on the same VNI. Two things are needed, both handled by [job-vae-core-extra.sh](examples/vae/script/job-vae-core-extra.sh):

1. `#SBATCH --network=single_node_vni,job_vni`. `job_vni` adds a job-wide VNI to every step (`SLINGSHOT_VNIS=<step VNI>,<job VNI>`); `single_node_vni` makes single-node steps (e.g. `--layout=split-node`) get a CXI service at all — without it `fi_domain()` fails with `-38 (Function not implemented)`.
2. In each task, keep only the job VNI: `export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}` before starting Python. libfabric's cxi provider uses only the first VNI listed, i.e. the per-step one, so without this the extra side's reads fail with `fi_cq_read ... prov_errno=25 (VNI_NOT_FOUND)`.

Verified on Frontier (2 nodes, split-node, `method=2`, `cxi`): with both, the extra step trains against the core step's data and both shut down cleanly; with either missing, it fails as above. MPI and RCCL inside each step work on the job VNI too.

### `get()` has no GPU destination-buffer pool

`DistDataset`/`DistDatasetReader`'s `get()` allocates a fresh GPU tensor per call on the GPU path (`--gpu-dest`), rather than reusing a pre-allocated pool. An earlier pooled design (round-robin slices of one pre-registered buffer, to amortize `fi_mr_regattr` cost) was removed after it was confirmed by direct experiment to corrupt data under `ThreadDataLoader` with `--num-workers > 1`: multiple worker threads raced for pool slots at per-sample granularity, and bounding how many batches could be in flight at once didn't bound which physical slots got overwritten, since slot-write order was determined by lock-acquisition order, not batch order. Removing the pool removes that race entirely — each call's destination is privately owned, nothing to reuse. The tradeoff: a fresh `fi_mr_regattr` whenever the new tensor isn't inside the previously registered range (the receive-MR cache in `read_from_remote()` reuses the registration when PyTorch's allocator hands back the same block, which is common for same-shape `torch.empty()`), instead of one registration shared across many. Revisit with a pool later if that registration cost matters (`--num-workers > 0` is otherwise known to work per the next section).

### Thread-safety of concurrent `get()` calls

`get()` is safe to call from multiple threads. For `method=1`/`2`, `DDStore::get()` serializes calls on the same variable with a per-variable mutex (`fabric_state::recv_lock` in `include/common.h`, taken via `fabric_state_lock_guard`), held for the whole RDMA read. It is required: `get()` writes per-variable fields (`recv_data`, the cached receive MR) that concurrent calls would otherwise race on, and the libfabric objects aren't opened thread-safe either (`hsn` requests `FI_THREAD_DOMAIN`, i.e. the application serializes access; `cxi` takes the provider's default from a NULL-hints `fi_getinfo()`) — without the lock, concurrent `get()` calls crashed with `double free or corruption`. Different variables have separate domains/endpoints/CQs and don't contend. `method=0` doesn't use the lock.

`get()` releases the GIL for the transfer on both the host and the GPU-destination path. The GPU-destination path first does a whole-device `torch.cuda.synchronize()` per call (see [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers)), and that sync still dominates: releasing the GIL there made no measurable difference to `--gpu-dest` epoch time. `add()` keeps the GIL (one-time collective setup).

### MPI thread level

DDStore and the examples use mpi4py's default initialization (`MPI_Init_thread` requesting `MPI_THREAD_MULTIPLE`). Only the main thread calls MPI — `add()`/`init()`/`join()` at setup, plus `epoch_begin()`/`epoch_end()` and `get()` for `method=0` — while `ThreadDataLoader` worker threads only call `get()` with `method=1`/`2`, which makes no MPI calls. So `MPI_THREAD_FUNNELED` is the minimum strictly required. On Frontier (Cray MPICH, 2 nodes × 8 ranks), `vae-ddp.py` and the pytest suites gave identical results and timing with `SINGLE`, `FUNNELED` and `MULTIPLE`. If you call MPI from your own worker threads with `method=0`, keep the default `MULTIPLE`.

### HIP streams and hardware queues (AMD/ROCm)

HIP maps streams onto a small pool of hardware queues per GPU per process — `GPU_MAX_HW_QUEUES`, 4 by default — and streams beyond that share a queue round-robin. Work in a shared queue runs in order, so a sync on a "separate" stream can still wait behind another stream's kernels. Measured on Frontier (ROCm 7.2): with the default stream kept busy, 12 of 16 new streams were independent of it by default (every 4th collided, including the first one created), 14 of 16 with `GPU_MAX_HW_QUEUES=8`, 15 of 16 with `16`. If you give data-loading threads their own streams, raise `GPU_MAX_HW_QUEUES` and remember the training stream and RCCL already occupy queues.

### GPU-to-GPU RDMA performance on AMD/ROCm

On Frontier, the GPU synchronization performed before each RDMA call (needed for correctness) can outweigh the benefit of skipping the host copy for small, per-sample transfers — GPU-to-GPU has not shown a speed advantage there yet, though results are correct either way. Larger, batched transfers should benefit more; that usage pattern isn't built yet.

### Troubleshooting: RDMA fails to connect (`cxi`, Frontier)

If `fi_domain()` fails with `-38 (Function not implemented)` on `cxi`, the step has no CXI service: add `#SBATCH --network=single_node_vni` — needed whenever a step runs on a single node (a `-N 1` job, or a one-node step inside a larger job). If ranks in different `srun` steps can't reach each other (`VNI_NOT_FOUND`), see [Multiple `srun` steps](#multiple-srun-steps-in-one-job-method2-cxi-frontier) above.

## Partitioned / Sub-communicator Usage

`PyDDStore` itself always spans the full communicator you pass it — there is no built-in "ranks per group" option. To run several independent stores side by side (e.g. one per node), split `comm` yourself before constructing `PyDDStore`, giving each group its own sub-communicator. Each group then holds a full replica of the dataset, partitioned across its own members.

**Example — 16 ranks split into groups of 4:**
```
ranks  0– 3  →  DDStore group 0
ranks  4– 7  →  DDStore group 1
ranks  8–11  →  DDStore group 2
ranks 12–15  →  DDStore group 3
```

This is useful when you want one store per node (e.g. 4 GPUs per node), limiting cross-node RDMA traffic to the dataset replication step at startup rather than every sample fetch.

```python
width = 4                                       # ranks per group, e.g. GPUs per node
sub_comm = comm.Split(rank // width, rank)
store = dds.PyDDStore(sub_comm)                  # one independent store per group
```

`DistDataset` in [examples/vae/distdataset.py](examples/vae/distdataset.py) wraps exactly this pattern behind a `ddstore_width` constructor argument — pass `ddstore_width=None` (default) for a single store across all ranks in `comm`, or an integer to split into groups of that size.

## PyTorch Dataset Integration

See [examples/vae/distdataset.py](examples/vae/distdataset.py) for a `torch.utils.data.Dataset` wrapper and [examples/vae/vae-ddp.py](examples/vae/vae-ddp.py) for a full DDP training example.

```bash
mpirun -n 4 python examples/vae/vae-ddp.py
```

`vae-ddp.py` and `vae_extra_train.py`/`vae_core_server.py` (the [method=2 split](#file-based-handshake-method2) variant) also accept `--gpu-dest`/`--gpu-source` to exercise [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers) end-to-end in a real training loop — `--gpu-dest` allocates the fetched batch directly on the training device, `--gpu-source` stores the local shard GPU-resident too:

```bash
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --gpu-dest --gpu-source
```

`--num-workers` (default 0) controls the training `DataLoader`'s parallelism. `0` uses PyTorch's normal `DataLoader`, single-threaded (no forked worker processes at all, so no MPI-after-`MPI_Init`-fork hazard). Any `--num-workers > 0` switches to [examples/vae/ddstore_dataloader.py](examples/vae/ddstore_dataloader.py)'s `ThreadDataLoader` instead — real threads, no fork, so it's safe together with `--gpu-dest`/`--gpu-source` too. The applied loader and worker count are printed at startup: `train_loader: DataLoader, num_workers=N` or `train_loader: ThreadDataLoader, num_workers=N`. `ThreadDataLoader` requires `DDSTORE_METHOD` 1 or 2 (libfabric) in `vae-ddp.py`. Concurrent `get()` calls from multiple threads are serialized inside `DDStore::get()` by a per-variable lock (see [Thread-safety](#thread-safety-of-concurrent-get-calls)), so threads gain overlap on everything except the RDMA call itself. `vae_extra_train.py` has the same `--num-workers` flag and behavior.

Measured with `vae-ddp.py`, `method=1`, `cxi`, 2 Frontier nodes × 8 ranks, `VAE_PROFILE=1`, average per-epoch time over epochs 2–8 (epochs are short, ~0.2 s, so treat as trends):

| `--num-workers` | host path: fetch / total (s) | `--gpu-dest --gpu-source`: fetch / total (s) |
|---|---|---|
| 0 (`DataLoader`) | 0.11 / 0.22 | 0.15 / 0.25 |
| 1 | 0.02 / 0.14–0.16 | 0.09 / 0.26 |
| 2 | 0.03 / 0.16 | 0.11 / 0.31 |
| 4 | 0.04 / 0.17 | 0.12 / 0.30 |
| 8 | 0.08 / 0.21 | 0.15 / 0.35–0.38 |

On the host path one worker thread (background prefetch) cuts epoch time ~30%; more workers make it steadily worse as they contend for the per-variable lock and the GIL. On the GPU path threading doesn't help, because of the per-call whole-device sync. All runs finished with the same final loss. Recommended: `--num-workers=1` on the host path, `0` on the GPU path.

These numbers use one `get()` per sample. With batched get (the default now; see [get_batch](#get_batchname-arr-indices) and the measurements under [Profiling](#profiling-get-ddstore_profile1)), every configuration is faster and the GPU path no longer suffers from workers.

The table predates moving batch collation into the worker thread (`ThreadDataLoader.fetch()`); with that change, the host path measured fetch ≈ 0.006 s / total ≈ 0.15 s per epoch with 1 worker and 0.044 / 0.19 with 2 (GPU path with 1 worker unchanged at 0.09 / 0.25).

```bash
# ThreadDataLoader, 1 worker thread, host path (no GPU buffers) -- the recommended setting
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1

# ThreadDataLoader + GPUDirect together
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1 --gpu-dest --gpu-source
```

### Slurm job scripts (Frontier)

[examples/vae/script/job-vae-single.sh](examples/vae/script/job-vae-single.sh) and [examples/vae/script/job-vae-core-extra.sh](examples/vae/script/job-vae-core-extra.sh) wrap the same VAE example for `sbatch` on Frontier — `job-vae-single.sh` runs plain DDP (one `srun` step), `job-vae-core-extra.sh` runs the [method=2 core/extra split](#file-based-handshake-method2) (two independent `srun` steps). Both take `--fabric`/`--gpudirect`; `job-vae-single.sh` also takes `--method` (the core/extra split is always `method=2` — `vae_core_server.py` sets it and the extra side always joins via method 2):

```bash
sbatch examples/vae/script/job-vae-single.sh                       # method=0, cxi
sbatch examples/vae/script/job-vae-single.sh --method=1 --gpudirect
sbatch examples/vae/script/job-vae-single.sh --method=1 --num-workers=4           # ThreadDataLoader, 4 worker threads
sbatch examples/vae/script/job-vae-single.sh --method=1 --gpudirect --num-workers=4  # ThreadDataLoader + GPUDirect
sbatch examples/vae/script/job-vae-core-extra.sh                   # method=2, cxi, colocate layout
sbatch examples/vae/script/job-vae-core-extra.sh --gpudirect --layout=split-node --core-nnodes=2
```

Run `--help` on either script for the full option list. `job-vae-single.sh` additionally has `--method` and `--num-workers` (default 0; `> 0` switches to `ThreadDataLoader`, see above). `job-vae-core-extra.sh` additionally has `--layout=colocate|split-node`, `--core-nnodes`, and `--num-workers` (for the extra/training step). Note `--layout=colocate` together with `--gpudirect` will over-request GPUs per node (core and extra each ask for a full node's worth of GPUs on the same nodes) — use `--layout=split-node` when testing GPUDirect on `job-vae-core-extra.sh`.

### Larger VAE cases

`vae-ddp.py` (and `vae_core_server.py` for the core/extra split; the extra side follows the published data) takes two size knobs, also exposed by both job scripts:

- `--replicate R` — repeat the MNIST training set `R` times: longer epochs, same per-sample cost.
- `--image-scale S` — upscale images to (28·S)×(28·S): each row is S² larger (3 KB at S=1, 12.5 KB at S=2); the VAE hidden layer grows to 400·S.

Both default to 1, which reproduces the original example exactly.

### Profiling `get()` (`DDSTORE_PROFILE=1`)

With `DDSTORE_PROFILE=1`, `PyDDStore.get_profile(name)` returns where `get()` time goes for a variable: lock wait, receive-MR check/registration (and miss count), posting `fi_read`, waiting for completion (C++, methods 1/2), plus the GPU-destination `torch.cuda.synchronize()` and whole-call time (Python). `vae-ddp.py` prints an all-rank summary at the end when it is set. Off by default.

[examples/scripts/bench_get.py](examples/scripts/bench_get.py) measures single-row `get()` latency/throughput vs row size, host vs GPU destination, and thread count, with the same breakdown.

Measured on Frontier (`method=1`, `cxi`, 2 nodes × 8 ranks):

- Inside the VAE, GPU-destination `get()` is dominated by the per-call device sync once a worker thread runs alongside training: ~4 µs per sample with no workers, but 130–190 µs with 1 worker and 290–430 µs with 2 (S=1/S=2), because the worker's sync waits for the training kernels. Lock wait (≤0.2 µs) and MR registration (<1 µs, even at 100% misses) are negligible; the RDMA itself is ~4–5 µs.
- `bench_get.py`, one thread, µs per single-row `get()`: 3 KB — host 8.6, GPU 19.5 (12.2 reusing the buffer); 12.5 KB — host 9.9, GPU 20.5; 200 KB — host 53, GPU 37; 1 MB — host 206–276, GPU 134. GPUDirect wins from somewhere between 12.5 KB and 200 KB per row; below that its fixed per-call overhead (sync, allocation) dominates.
- A second thread adds no per-rank throughput: the per-variable lock serializes the transfers (lock wait ≈ transfer time at large rows).

With `get_batch()` (`bench_get.py --batch 128`, µs per row, 1 thread): 3 KB — host 0.68, GPU 0.78 (from 9.0 / 20.9 with one row per call); 12.5 KB — host 1.82, GPU 1.38; 200 KB — host 27.5, GPU 19.6; 1 MB — host 189, GPU 99 (~10.6 GB/s per rank). The GPU sync drops to ~0.06 µs per row, and GPUDirect now beats host from 12.5 KB rows up. (Host-destination batches of 1 MB rows are slower than single reads; not investigated.)

VAE (`vae-ddp.py`, epochs 2–8 average, s/epoch), per-sample `get()` → batched (`DDSTORE_BATCH_GET` 0 → 1); losses identical (8.8960 at S=1, 30.0178 at S=2):

| `--image-scale` | host, 0 workers | host, 1 worker | GPU, 0 workers | GPU, 1 worker | GPU, 2 workers |
|---|---|---|---|---|---|
| 1 | 0.247 → 0.132 | 0.156 → 0.126 | 0.263 → 0.135 | 0.259 → 0.127 | 0.403 → 0.134 |
| 2 | 0.395 → 0.286 | 0.293 → 0.265 | 0.428 → 0.277 | 0.416 → 0.264 | 0.518 → 0.297 |

The per-row GPU sync falls from 129–383 µs (with workers) to ~0.1 µs, and training compute time recovers because it no longer waits behind the workers' syncs.

## Testing

### Unit tests (pytest)

Install test dependencies:

```bash
pip install pytest pytest-mpi
```

**Single-rank** — no cluster required, covers all dtypes, `add`/`get`/`init`/`update`, and error cases:

```bash
mpirun -n 1 python -m pytest test/test_single.py -v
```

**Multi-rank** — verifies remote reads across all rank pairs and sub-communicator grouping:

```bash
mpirun -n 4 python -m pytest test/test_multirank.py -v
```

**GPUDirect RDMA** — requires a live `cxi` fabric and a CUDA/HIP GPU per rank (skipped automatically otherwise); see [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers):

```bash
DDSTORE_FABRIC=cxi mpirun -n 2 python -m pytest test/test_gpu_rdma.py -v
```

| Test file | Min ranks | What is tested |
|---|---|---|
| `test/test_single.py` | 1 | All dtypes, `add`/`get`, `init`/`update`/`get`, error handling, double `free()` |
| `test/test_multirank.py` | 2 (4 recommended) | Remote reads, shard boundaries, multiple variables, `ddstore_width` grouping |
| `test/test_gpu_rdma.py` | 2 | GPU-resident `add()`/`get()` in both directions, both libfabric methods, negative/error cases |
| `test/test_get_batch.py` | 2 (4 recommended) | `get_batch()`: shuffled indices across ranks with repeats, single row, dtypes, error recovery, GPU destination, concurrent threads; method 0, plus method 1 over `cxi` inside a Slurm step |

### Integration scripts

```bash
# Basic functional test (libfabric, method=1)
mpirun -n 4 python examples/scripts/demo.py

# Integration test with PyTorch DDP (libfabric, method=1)
mpirun -n 4 python examples/scripts/test.py
```

Optional arguments for `examples/scripts/demo.py` and `examples/scripts/test.py`:

| Flag | Default | Description |
|---|---|---|
| `--num` | `1048576` | Rows per rank |
| `--dim` | `64` | Elements per row |
| `--nbatch` | `32` | Number of random reads |
| `--gloo` / `--nccl` | `--gloo` | `test.py` only: `torch.distributed` backend |

### Method 2 (file-based handshake)

Two separate launches sharing a handshake directory on a shared filesystem — not a single `mpirun`, since core and extra are independent jobs:

```bash
# Terminal 1 — core (data-holding) side
mpirun -n 4 python test/test_method2_core.py /path/to/shared/ddstore_hs

# Terminal 2 — extra (reader) side, after or while the core side is running
python test/test_method2_extra.py /path/to/shared/ddstore_hs 4
```

## Citation

If you use DDStore in your research, please cite:

```bibtex
@inproceedings{choi2023ddstore,
  title={DDStore: Distributed data store for scalable training of graph neural networks on large atomistic modeling datasets},
  author={Choi, Jong Youl and Lupo Pasini, Massimiliano and Zhang, Pei and Mehta, Kshitij and Liu, Frank and Bae, Jonghyun and Ibrahim, Khaled},
  booktitle={Proceedings of the SC'23 Workshops of the International Conference on High Performance Computing, Network, Storage, and Analysis},
  pages={941--950},
  year={2023}
}
```

## License

See [LICENSE](LICENSE).
