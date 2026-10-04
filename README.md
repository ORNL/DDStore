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

For `method=1`/`2` the whole batch is one call: one lock acquisition, one memory registration, and (GPU destination) one device sync, with all of the batch's `fi_read`s posted before any is waited for, so the reads overlap on the network. It is still one `fi_read` per row.

For `method=0`, `get_batch()` is **collective**, after the collective module of [MDLoader](https://ieeexplore.ieee.org/abstract/document/10820758) (see [Citation](#citation)): every rank all-gathers all ranks' indices (`MPI_Allgatherv`), packs the rows it owns for each requester, and one `MPI_Alltoallv` delivers them, on a private duplicate of the store's communicator, in rounds of at most `DDSTORE_ALLTOALL_MAX_BYTES` (default 2 MiB) received per rank so large rows don't turn into one huge exchange. So every rank must call it for the variable the same number of times, in the same order, from one thread at a time; the number of indices may differ per rank (including 0). Indices are checked on the gathered list, so a bad index raises on every rank together. `DistributedSampler` gives every rank the same number of batches, and `vae-ddp.py` allows no worker threads with `method=0`, so the data loaders meet this automatically.

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

## Environment variables

**Read by DDStore itself** (the C++ library, `pyddstore`, `cpu_nic_map`):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_FABRIC` | `hsn` | libfabric provider for `method=1`/`2`: `hsn` (`tcp;ofi_rxm`) or `cxi` (native Slingshot; required for [GPUDirect RDMA](#gpudirect-rdma-gpu-resident-buffers)). See [libfabric RDMA](#libfabric-rdma-method1). |
| `FABRIC_IFACE` | auto | Network interface (libfabric domain, e.g. `cxi0`, `hsn0`) for `method=1`/`2`. Set it to force one; otherwise picked from the rank's CPU affinity. |
| `DDSTORE_NIC_MAP` | unset | Precomputed CPU→NIC map used for that automatic pick instead of a live hwloc query (`python3 -m cpu_nic_map --env`). The constructor's `nic_map=` argument takes priority. |
| `DDSTORE_HANDSHAKE_DIR` | `./ddstore_hs` | `method=2` handshake directory when none is given (C++ API; `PyDDStore` requires `handshake_dir`, and the examples fill it from this variable). Must be on a shared filesystem. |
| `DDSTORE_HANDSHAKE_TIMEOUT_S` | `300` | Seconds a `method=2` extra member's `join()` polls for the core group's record file. |
| `DDSTORE_PROFILE` | off | `1` turns on `get()`/`get_batch()` timing counters, read with `get_profile(name)`. See [Performance](#performance). |
| `DDSTORE_ALLTOALL_MAX_BYTES` | `2097152` (2 MiB) | `method=0` `get_batch()`: bytes each rank receives per exchange round. Must be equal on all ranks. |

The backend itself is not an environment variable in the library: pass `method=` to `PyDDStore` (`DDSTORE_METHOD` below is how the examples choose it).

**Read by the examples** (`examples/vae/`, `examples/scripts/`, job scripts):

| Variable | Default | Effect |
|---|---|---|
| `DDSTORE_METHOD` | `0` (`bench_get.py`: `1`) | Backend passed as `method=`: `0` MPI RMA, `1` libfabric, `2` file-based handshake. `--num-workers > 0` in `vae-ddp.py` needs `1` or `2`. |
| `DDSTORE_BATCH_GET` | `1` | `DistDataset.__getitems__` reads a whole batch with one `get_batch()`; `0` falls back to one `get()` per sample. |
| `DDSTORE_N_CORE` | `4` | `vae_extra_train.py`, `test_method2_*.py`: number of core ranks that published the data (`--n-core` overrides). |
| `DDSTORE_AFFINITY_WIDTH` / `DDSTORE_AFFINITY_OFFSET` | `0` / `0` | `ThreadDataLoader`: pin worker thread *i* to CPUs `[offset + i·width, offset + (i+1)·width)` of the process's affinity; width `0` = no pinning. |
| `DDSTORE_BACKEND` | auto | `torch.distributed` backend for the examples' DDP setup (`nccl`, `gloo`, `xccl`). |
| `VAE_PROFILE` | off | `1`: `vae-ddp.py` prints per-epoch fetch vs compute time. |
| `MASTER_PORT` | `2345` | DDP rendezvous port; the core/extra job script gives each step its own. |

**System settings that matter on Frontier:**

| Variable | Effect |
|---|---|
| `SLINGSHOT_VNIS` | Set by Slurm per step. With `--network=job_vni`, keep only the last (job-wide) entry before starting Python so separate `srun` steps can reach each other — see [Multiple `srun` steps](#multiple-srun-steps-in-one-job-method2-cxi). |
| `GPU_MAX_HW_QUEUES` | ROCm hardware queues per GPU per process (default 4); raise it if data-loading threads use their own streams — see [HIP streams](docs/results.md#hip-streams-and-hardware-queues-frontier-rocm-72). |

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

Environment variables: `DDSTORE_HANDSHAKE_DIR`, `DDSTORE_HANDSHAKE_TIMEOUT_S`, `DDSTORE_FABRIC` and `DDSTORE_NIC_MAP` — see [Environment variables](#environment-variables).

See [test/test_method2_core.py](test/test_method2_core.py) / [test/test_method2_extra.py](test/test_method2_extra.py) for a minimal runnable pair, and [examples/vae/vae_core_server.py](examples/vae/vae_core_server.py) / [examples/vae/vae_extra_train.py](examples/vae/vae_extra_train.py) for a full DDP training example using this split.

`ddstore_width` grouping (below) is not currently supported with `method=2` — every core rank in `comm` is treated as one group.

## GPUDirect RDMA (GPU-resident buffers)

`add()`, `get()` and `get_batch()` accept a CUDA/HIP `torch.Tensor` in place of a NumPy array, so RDMA reads from or writes directly into GPU memory, with no `.cpu()`/`.to(device)` copy. Requires `method=1` or `2`, **`DDSTORE_FABRIC=cxi`**, and a CUDA- or ROCm-enabled PyTorch. A GPU tensor with `DDSTORE_FABRIC=hsn` (the default) or `method=0` raises a clear error instead of silently copying through the host.

```python
import torch
data = torch.rand(1024, 64, dtype=torch.float32, device="cuda")
store.add("features", data)                      # GPU source, no host copy

out = torch.empty((1, 64), dtype=torch.float32, device="cuda")
store.get("features", out, start=2048)            # GPU destination, no host copy
```

- **`add()` with a GPU tensor registers your tensor's own memory; no copy is made.** Keep it alive and unmodified until `free()`. `PyDDStore` holds a reference as a safety net, and adding the same name again with a GPU tensor is rejected. (With NumPy, `add()` copies and the array can be reused right away.)
- **The device is synchronized before each GPU transfer** (`torch.cuda.synchronize()`, once per `get()` / `get_batch()` call): the NIC writes outside PyTorch's stream ordering, and without the sync training hit GPU memory faults. Prefer `get_batch()` on the GPU path so this costs one sync per batch, not per sample.
- `init()`/`update()` stay host-only.
- Whether GPU destinations are faster than host ones depends on the machine: on Frontier they win from ~12.5 KB rows up, on Perlmutter host destinations win at every size ([results](docs/results.md#bench_getpy-µs-per-row)).

Examples: [test/test_gpu_rdma.py](test/test_gpu_rdma.py), and `--gpu-dest`/`--gpu-source` on [vae-ddp.py](examples/vae/vae-ddp.py), [vae_extra_train.py](examples/vae/vae_extra_train.py) and [vae_core_server.py](examples/vae/vae_core_server.py).

## PyTorch Dataset Integration

[examples/vae/distdataset.py](examples/vae/distdataset.py) wraps a store as a `torch.utils.data.Dataset`, and [examples/vae/vae-ddp.py](examples/vae/vae-ddp.py) trains a VAE with DDP on top of it:

```bash
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1
DDSTORE_METHOD=1 DDSTORE_FABRIC=cxi mpirun -n 4 python examples/vae/vae-ddp.py --num-workers=1 --gpu-dest --gpu-source
```

- **Batched by default.** `DistDataset.__getitems__` reads each training batch with one [`get_batch()`](#get_batchname-arr-indices) call; `DDSTORE_BATCH_GET=0` falls back to one `get()` per sample.
- **`--num-workers`** (default 0): `0` uses PyTorch's standard `DataLoader` in the main process; `> 0` uses [`ThreadDataLoader`](examples/vae/ddstore_dataloader.py), which fetches and collates batches in worker *threads* (no fork, so it is safe with MPI and GPU buffers). It needs `method=1`/`2`. One or two workers are enough: reads on one variable are serialized by its lock, and a single batched call already keeps the network busy.
- **`--gpu-dest` / `--gpu-source`**: fetched batches land directly on the training GPU / each rank's shard is stored on its GPU (see [GPUDirect](#gpudirect-rdma-gpu-resident-buffers)).
- **`--replicate R`** repeats the training set R times (longer epochs); **`--image-scale S`** upscales images to (28·S)² so each row is S² larger. Both default to 1, the original example.
- The [method=2 split](#file-based-handshake-method2) variant is [vae_core_server.py](examples/vae/vae_core_server.py) (holds the data) + [vae_extra_train.py](examples/vae/vae_extra_train.py) (trains), with the same options.

### Slurm job scripts

[job-vae-single.sh](examples/vae/script/job-vae-single.sh) runs `vae-ddp.py` as one `srun` step; [job-vae-core-extra.sh](examples/vae/script/job-vae-core-extra.sh) runs the core/extra split as two steps. Run either with `--help` for all options. Their `#SBATCH` lines target Frontier (`-A FUS184`, 8 ranks × 7 cores per node); adjust for other machines.

```bash
sbatch examples/vae/script/job-vae-single.sh --method=1 --num-workers=1
sbatch examples/vae/script/job-vae-single.sh --method=1 --gpudirect --image-scale=2
sbatch examples/vae/script/job-vae-core-extra.sh                       # split-node: 1 core node, the rest extra
sbatch examples/vae/script/job-vae-core-extra.sh --gpudirect --core-nnodes=2
```

`job-vae-core-extra.sh` sets up Slingshot networking for its two steps (see [Multiple `srun` steps](#multiple-srun-steps-in-one-job-method2-cxi)). `--layout=colocate` (both steps on the same nodes) works on Perlmutter only. On Perlmutter, [examples/scripts/perlmutter-check.sh](examples/scripts/perlmutter-check.sh) runs the whole check-list in one job ([docs/perlmutter-checklist.md](docs/perlmutter-checklist.md)).

## Partitioned / Sub-communicator Usage

`PyDDStore` always spans the whole communicator you pass it. To run several independent stores side by side (e.g. one per node), split `comm` first; each group then holds a full replica of the dataset, partitioned across its own members:

```python
width = 4                                       # ranks per group, e.g. GPUs per node
sub_comm = comm.Split(rank // width, rank)
store = dds.PyDDStore(sub_comm)                  # one independent store per group
```

This keeps sample fetches inside a node at the cost of replicating the data per group. `DistDataset` exposes it as `ddstore_width` (`None` = one store across all ranks). Not supported with `method=2`.

## Performance

- Use **batched reads** (the default with `DistDataset`, or `get_batch()` directly). They cut per-sample cost by 10–27× for small rows and make the GPU path insensitive to worker threads; in the VAE every configuration got 1.2–3.9× faster per epoch.
- `method=1` (one-sided `fi_read`) is the fastest backend; `method=0` with batching (collective) comes close for small rows.
- `DDSTORE_PROFILE=1` + `get_profile(name)` shows where `get()`/`get_batch()` time goes: lock wait, memory registration, posting and completing `fi_read`, GPU sync. `vae-ddp.py` prints an all-rank summary when it is set. [examples/scripts/bench_get.py](examples/scripts/bench_get.py) measures per-row latency and throughput vs row size, destination, batch size and threads.

Measurements, profiles and the experiments behind these choices: [docs/results.md](docs/results.md).

## Known Limitations

### Multiple `srun` steps in one job (`method=2`, `cxi`)

On Slingshot every `srun` step gets its own VNI (network isolation ID), and two endpoints can only talk on the same VNI. For core and extra running as separate steps, [job-vae-core-extra.sh](examples/vae/script/job-vae-core-extra.sh) does both of these:

1. `#SBATCH --network=single_node_vni,job_vni`: `job_vni` adds a job-wide VNI to every step (`SLINGSHOT_VNIS=<step VNI>,<job VNI>`); on Frontier, `single_node_vni` is also what gives single-node steps a CXI service at all (without it `fi_domain()` fails with `-38`).
2. In each task, before Python starts: `export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}`. libfabric's cxi provider uses only the first VNI listed (the step's own), so without this reads fail with `VNI_NOT_FOUND`.

Colocating both steps on the same nodes:

| | `job_vni` + wrapper | `job_vni` + wrapper + `srun --overlap` | no `--network` |
|---|---|---|---|
| Frontier | second step fails to launch: `Error configuring interconnect` | same failure | each step has only its own VNI: extra cannot reach core |
| Perlmutter | second step does not start | **works** | extra cannot reach core |

So the script defaults to `--layout=split-node`; `--layout=colocate` (with `--overlap`) is for Perlmutter. Perlmutter's single-node steps also work without the `--network` flags.

### Concurrency

- `get()` and `get_batch()` are thread-safe. For `method=1`/`2` a per-variable lock in `DDStore::get()` serializes calls on one variable (it guards shared receive state; without it concurrent calls crashed). Both release the GIL during the transfer.
- `method=0` `get_batch()` is **collective**: every rank calls it the same number of times, in the same order, from one thread. `vae-ddp.py` therefore allows no worker threads with `method=0`.
- Only the main thread calls MPI (setup, `epoch_begin`/`epoch_end`, `method=0` reads); mpi4py's default `MPI_THREAD_MULTIPLE` is fine, `FUNNELED` is the minimum. If you call MPI from your own worker threads, keep `MULTIPLE`.

### Troubleshooting: RDMA fails to connect (`cxi`)

If `fi_domain()` fails with `-38 (Function not implemented)`, the step has no CXI service: add `#SBATCH --network=single_node_vni` (needed on Frontier for any single-node step, i.e. a `-N 1` job or a one-node step inside a larger job). If ranks in different `srun` steps can't reach each other (`VNI_NOT_FOUND`), see [Multiple `srun` steps](#multiple-srun-steps-in-one-job-method2-cxi) above.

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

```bibtex
@inproceedings{bae2024mdloader,
  title={MDLoader: A Hybrid Model-Driven Data Loader for Distributed Graph Neural Network Training},
  author={Bae, Jonghyun and Choi, Jong Youl and Lupo Pasini, Massimiliano and Mehta, Kshitij and Zhang, Pei and Ibrahim, Khaled},
  booktitle={SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2024},
  month={nov},
  doi={10.1109/SCW63240.2024.00145}
}
```

## License

See [LICENSE](LICENSE).
