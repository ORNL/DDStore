# Installation

## Prerequisites

| Dependency | Notes |
|---|---|
| MPI (OpenMPI / MPICH) | `mpicc` and `mpicxx` must be on `PATH`. Optional: see [Building without MPI](#building-without-mpi) |
| libfabric | Required for the RDMA backends (`method=1` and `method=2`) |
| Python ≥ 3.9 | |
| NumPy, Cython | Python build dependencies |
| mpi4py | Runtime dependency of an MPI build (installed with it); not needed to build |
| PyTorch (optional) | For `pyddstore.torch` and GPU buffers (CUDA or ROCm build) |

## Building and installing

```bash
# Install Python build dependencies
pip install numpy mpi4py Cython

# Build in-place (use with PYTHONPATH=$PWD/src:$PYTHONPATH)
CC=mpicc CXX=mpicxx python setup.py build_ext --inplace

# Or install into the active virtual environment
CC=mpicc CXX=mpicxx pip install .
CC=mpicc CXX=mpicxx pip install ".[torch]"     # also pulls PyTorch, for pyddstore.torch

# Or install in editable/development mode
CC=mpicc CXX=mpicxx pip install -e .

# Or install directly from GitHub
CC=mpicc CXX=mpicxx pip install git+https://github.com/ORNL/DDStore.git
```

To build against the environment's own packages (e.g. an `mpi4py` built for Cray MPICH), disable build isolation:

```bash
CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
```

If that fails with `No module named 'distutils.msvccompiler'` (new setuptools with an old system NumPy, e.g. `cray-python/3.11.7` on Frontier), use the standard-library `distutils`:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
```

## Building without MPI

`DDSTORE_MPI=0` builds without MPI (no `mpi.h`, library or mpi4py), with a plain compiler:

```bash
DDSTORE_MPI=0 CC=gcc CXX=g++ pip install .
```

Methods 1 and 2 only; ranks exchange records through files ([Without MPI](backends.md#without-mpi-methods-1-and-2)) and are launched with `srun`. `pyddstore.HAVE_MPI` is `False`; a communicator or `method=0` raises. Both builds share the generated `_core.cpp`, so switching only rebuilds the extension.

## Notes

- Package `pyddstore` (compiled `pyddstore._core`, plus `pyddstore.torch`). After updating from 1.x, rebuild; leftover `src/pyddstore.cpython-*.so` / `src/pyddstore.cpp` can be deleted.
- `src/pyddstore/_core.cpp` is shared by every environment building from the checkout; `setup.py` regenerates it when the NumPy major version changes.
