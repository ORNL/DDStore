# DDStore

<img src="images/DDStore-logo.png" alt="DDStore logo" />

Efficient distributed data loading for distributed data-parallel (DDP) training.

Each MPI rank holds a shard of the full dataset in memory. DDStore exposes a global index space so any rank can read any sample via one-sided remote memory access — either MPI RMA (default) or libfabric RDMA — without coordinator synchronization.

- **Batched reads**: `get_batch()` fetches a whole training batch in one call (one-sided RDMA reads in flight together, or an MPI collective for `method=0`).
- **GPUDirect RDMA**: data can live in, and be read straight into, GPU memory.
- **PyTorch integration**: `pyddstore.torch` turns any map-style dataset into a distributed one (`DistDataset`), reads samples made of several stored rows (`WindowedDataset`), and provides a thread-based `ThreadDataLoader` that is safe with MPI and GPU buffers.
- **Thread-safe** reads, a profiler for where read time goes, and a split mode (`method=2`) where a separate job reads data published by another.

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="DDStore architecture" width="300" />

**Documentation: <https://ornl.github.io/DDStore/>** (source in [docs/](docs/index.md)).

## Prerequisites

| Dependency | Notes |
|---|---|
| MPI (OpenMPI / MPICH) | `mpicc` and `mpicxx` must be on `PATH` |
| libfabric | Required for the RDMA backends (`method=1` and `method=2`) |
| Python ≥ 3.9 | |
| NumPy, mpi4py, Cython | Python build dependencies |
| PyTorch (optional) | For `pyddstore.torch` and GPU buffers (CUDA or ROCm build) |

## Installation

```bash
pip install numpy mpi4py Cython
CC=mpicc CXX=mpicxx pip install .              # or ".[torch]" to also pull PyTorch
CC=mpicc CXX=mpicxx pip install -e .           # editable, for development
```

On Cray systems, building against the environment's own `mpi4py` (`--no-build-isolation`), and other build details: [Installation](docs/installation.md).

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
