# DDStore

<img src="images/DDStore-logo.png" alt="DDStore logo" />

Efficient distributed data loading for distributed data-parallel (DDP) training.

Each rank holds a shard of the dataset in memory; any rank reads any sample by global index through one-sided remote memory access (MPI RMA or libfabric RDMA), without coordination.

- **Batched reads**: `get_batch()` reads a whole batch in one call.
- **GPUDirect RDMA**: data can live in, and be read straight into, GPU memory.
- **PyTorch integration**: `DistDataset` (any map-style dataset), `DistStore` (several datasets on one store), `WindowedDataset` (multi-row samples), `ThreadDataLoader` (threads; safe with MPI and GPU buffers). PyTorch's `DataLoader` with forked workers also works (methods 1 and 2).
- **Thread-safe** reads, a read-time profiler, and a split mode (`method=2`) where one job reads data published by another.
- **MPI optional**: methods 1 and 2 also run without MPI (`DDSTORE_MPI=0`), launched by `srun`.

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="DDStore architecture" width="300" />

**Documentation: <https://ornl.github.io/DDStore/>** (source in [docs/](docs/index.md)).

## Prerequisites

| Dependency | Notes |
|---|---|
| MPI (OpenMPI / MPICH) | `mpicc` and `mpicxx` must be on `PATH`. Optional: `DDSTORE_MPI=0` builds without it (methods 1 and 2) |
| libfabric | Required for the RDMA backends (`method=1` and `method=2`) |
| Python ≥ 3.9 | |
| NumPy, Cython | Python build dependencies (mpi4py too for an MPI build) |
| PyTorch (optional) | For `pyddstore.torch` and GPU buffers (CUDA or ROCm build) |

## Installation

```bash
pip install numpy mpi4py Cython
CC=mpicc CXX=mpicxx pip install .              # or ".[torch]" to also pull PyTorch
CC=mpicc CXX=mpicxx pip install -e .           # editable, for development
DDSTORE_MPI=0 CC=gcc CXX=g++ pip install .     # without MPI: methods 1 and 2 only
```

Cray systems and other build details: [Installation](docs/installation.md).

## Quick start

```python
import numpy as np
from mpi4py import MPI
import pyddstore as dds

comm = MPI.COMM_WORLD
store = dds.PyDDStore(comm)                  # MPI RMA; method=1 for libfabric RDMA

data = np.random.rand(1024, 64).astype(np.float32)
store.add("features", data)                  # collective: each rank adds its shard

out = np.zeros((1, 64), dtype=np.float32)
store.epoch_begin()
store.get("features", out, start=2048)       # any global row, from any rank
store.epoch_end()
store.free()
```

With PyTorch:

```python
import torch                                 # import torch before MPI starts
from mpi4py import MPI
from pyddstore.torch import DistDataset, ThreadDataLoader

trainset = DistDataset(my_dataset, "train", MPI.COMM_WORLD)   # each rank loads only its share
sampler = torch.utils.data.distributed.DistributedSampler(trainset)
loader = ThreadDataLoader(trainset, batch_size=128, sampler=sampler, num_workers=1)
for x, y in loader:
    ...
```

Run with `mpirun -n 4 python my_script.py` (or `srun`).

PyTorch's `DataLoader` with forked workers also works (methods 1 and 2): set `DDSTORE_FORK_WORKERS=1`, pass `persistent_workers=True`, and with MPI import pyddstore before mpi4py. Workers read into host memory; use `ThreadDataLoader` for GPU destinations. See [Forked DataLoader workers](docs/pytorch.md#forked-dataloader-workers).

Without MPI (`DDSTORE_MPI=0` build or `DDSTORE_COMM=file`), records are exchanged through files in a shared directory; ranks come from `srun`:

```python
import os
import pyddstore as dds
from pyddstore.torch import DistDataset

hs = "/lustre/.../hs"                        # on a filesystem every rank sees
store = dds.PyDDStore(None, method=1, handshake_dir=hs,
                      rank=int(os.environ["SLURM_PROCID"]), size=int(os.environ["SLURM_NTASKS"]))
trainset = DistDataset(my_dataset, "train", handshake_dir=hs)   # comm=None, method 1
```

See [Without MPI](docs/backends.md#without-mpi-methods-1-and-2).

## Documentation

| | |
|---|---|
| Getting started | [Installation](docs/installation.md), [Quick start](docs/quickstart.md) |
| User guide | [Backends](docs/backends.md) (MPI RMA, libfabric, file-based handshake, partitioned stores), [GPUDirect RDMA](docs/gpudirect.md), [PyTorch integration](docs/pytorch.md), [HPC systems](docs/hpc.md) (Slurm, Slingshot, Frontier, Perlmutter), [Performance](docs/performance.md), [Concurrency](docs/concurrency.md) |
| Reference | [`PyDDStore`](docs/api-pyddstore.md), [`pyddstore.torch`](docs/api-torch.md), [Environment variables](docs/environment.md) |
| More | [Testing](docs/testing.md), [Measurements](docs/results.md) |

To build the documentation locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
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
