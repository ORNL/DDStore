# DDStore

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

## Quick Start

```python
import mpi4py
mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

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

### `PyDDStore(comm_or_none=None, method=0, handshake_dir="", n_core=0)`

| Parameter | Type | Description |
|---|---|---|
| `comm_or_none` | `mpi4py.MPI.Comm` or `None` | MPI communicator covering all ranks. `None` only for a `method=2` extra member |
| `method` | `int` | `0` = MPI RMA (default), `1` = libfabric RDMA, `2` = file-based handshake (see [below](#file-based-handshake-method2)) |
| `handshake_dir` | `str` | Required for `method=2`: shared-filesystem directory used to exchange RDMA addresses |
| `n_core` | `int` | Required for a `method=2` extra member: number of core ranks that published data |

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
| `arr` | `np.ndarray` | C-contiguous 2-D (or 1-D) array. Supported dtypes: `int32`, `int64`, `uint8`, `float32`, `float64`, `bool_` |

---

### `update(name, arr, offset=0)`

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
| `arr` | `np.ndarray` | Pre-allocated, C-contiguous output buffer |
| `start` | `int` | Global row index |

---

### `join(name)`

`method=2` extra member only. Discovers a variable published by the core group by polling the handshake directory until every core rank's record file appears (up to `DDSTORE_HANDSHAKE_TIMEOUT_S` seconds), then registers it for `get()`.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier, matching the `name` used in the core group's `add()` |

---

### `info(name)`

Returns `(total_rows, disp, itemsize)` for a variable that has been `add()`-ed or `join()`-ed. Useful on the extra side to size output buffers without hardcoding shapes.

---

### `epoch_begin()` / `epoch_end()`

Open and close an MPI RMA access epoch (calls `MPI_Win_fence`). **Collective**. Required around `get()` calls when using `method=0`. No-op for `method=1`.

---

### `free()`

Release all MPI windows and allocated memory. Safe to call after `MPI_Finalize`.

## Backends

### MPI RMA (`method=0`, default)

Uses `MPI_Win_create` and `MPI_Get` for one-sided remote reads. Works on any MPI-capable cluster without additional hardware. `epoch_begin`/`epoch_end` are required to delimit access epochs.

### libfabric RDMA (`method=1`)

Uses `fi_read` for true RDMA transfers over high-speed interconnects (Infiniband/verbs, Cray GNI, Intel PSM2). Lower latency than MPI RMA on supported hardware. `epoch_begin`/`epoch_end` are no-ops with this backend.

Set `FABRIC_IFACE` to select a specific network interface when the automatic selection picks the wrong one:
```bash
export FABRIC_IFACE=hsn0   # e.g. Cray Slingshot
```

### File-based handshake (`method=2`)

Splits the dataset-holding job from the training job entirely: a **core** group loads and publishes data, and a separate **extra** group reads it over RDMA (`fi_read`, same transport as `method=1`) — the two are independent MPI jobs (e.g. two separate `srun`/`mpirun` launches, possibly on different node allocations) that never share a communicator. They rendezvous only through record files written to a shared-filesystem directory (must be visible to all nodes, e.g. Lustre):

- **Core member** — has an MPI communicator, publishes with `add()`/`init()`. Each rank writes a `{name}_rank{N}.bin` record (fabric address, MR key, base pointer, row count, dtype) into `handshake_dir`.
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

See [test/test_method2_core.py](test/test_method2_core.py) / [test/test_method2_extra.py](test/test_method2_extra.py) for a minimal runnable pair, and [examples/vae/vae_core_server.py](examples/vae/vae_core_server.py) / [examples/vae/vae_extra_train.py](examples/vae/vae_extra_train.py) for a full DDP training example using this split.

`ddstore_width` grouping (below) is not currently supported with `method=2` — every core rank in `comm` is treated as one group.

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

| Test file | Min ranks | What is tested |
|---|---|---|
| `test/test_single.py` | 1 | All dtypes, `add`/`get`, `init`/`update`/`get`, error handling, double `free()` |
| `test/test_multirank.py` | 2 (4 recommended) | Remote reads, shard boundaries, multiple variables, `ddstore_width` grouping |

### Integration scripts

```bash
# Basic functional test (MPI RMA)
mpirun -n 4 python examples/scripts/demo.py

# Integration test with PyTorch DDP
mpirun -n 4 python examples/scripts/test.py
```

Optional arguments for `examples/scripts/demo.py` and `examples/scripts/test.py`:

| Flag | Default | Description |
|---|---|---|
| `--num` | `1048576` | Rows per rank |
| `--dim` | `64` | Elements per row |
| `--nbatch` | `32` | Number of random reads |

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
