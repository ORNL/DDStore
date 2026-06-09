# DDStore

Efficient distributed data loading for distributed data-parallel (DDP) training.

Each MPI rank holds a shard of the full dataset in memory. DDStore exposes a global index space so any rank can read any sample via one-sided remote memory access — either MPI RMA (default) or libfabric RDMA — without coordinator synchronization.

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="DDStore architecture" width="300" />

## Prerequisites

| Dependency | Notes |
|---|---|
| MPI (OpenMPI / MPICH) | `mpicc` and `mpicxx` must be on `PATH` |
| libfabric | Required for RDMA backend (`method=1`) |
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

### `PyDDStore(comm, method=0)`

| Parameter | Type | Description |
|---|---|---|
| `comm` | `mpi4py.MPI.Comm` | MPI communicator for the store group |
| `method` | `int` | `0` = MPI RMA (default), `1` = libfabric RDMA |

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

## Partitioned / Sub-communicator Usage

To partition a large job into independent DDStore groups (e.g. one store per node):

```python
ddstore_width = 8   # ranks per store group
ddstore_comm = comm.Split(rank // ddstore_width, rank)
store = dds.PyDDStore(ddstore_comm)
```

## PyTorch Dataset Integration

See [examples/vae/distdataset.py](examples/vae/distdataset.py) for a `torch.utils.data.Dataset` wrapper and [examples/vae/vae-ddp.py](examples/vae/vae-ddp.py) for a full DDP training example.

```bash
mpirun -n 4 python examples/vae/vae-ddp.py
```

## Testing

```bash
# Basic functional test (MPI RMA)
mpirun -n 4 python test/demo.py

# Integration test with PyTorch DDP
mpirun -n 4 python test/test.py
```

Optional arguments for `test/demo.py` and `test/test.py`:

| Flag | Default | Description |
|---|---|---|
| `--num` | `1048576` | Rows per rank |
| `--dim` | `64` | Elements per row |
| `--nbatch` | `32` | Number of random reads |

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
