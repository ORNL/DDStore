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

To build against the packages already in the current environment (e.g. an `mpi4py` built against Cray MPICH) instead of letting pip fetch fresh build dependencies into an isolated build environment, disable build isolation:

```bash
CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
```

If that fails with `ModuleNotFoundError: No module named 'distutils.msvccompiler'` (newer setuptools combined with an older system NumPy, e.g. `cray-python/3.11.7` on Frontier), point setuptools at the standard-library `distutils` for the build:

```bash
SETUPTOOLS_USE_DISTUTILS=stdlib CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
```

## Building without MPI

`DDSTORE_MPI=0` builds DDStore with no MPI at all: no `mpi.h`, no MPI library, no mpi4py. Use a plain compiler:

```bash
DDSTORE_MPI=0 CC=gcc CXX=g++ pip install .
```

Such a build has methods 1 and 2 only. Stores are made without a communicator, and the ranks exchange their RDMA records through files in a shared directory instead of `MPI_Allgather` (see [Without MPI](backends.md#without-mpi-methods-1-and-2)). `pyddstore.HAVE_MPI` is `False`; passing a communicator, or `method=0`, raises. Launch the ranks with `srun`: each one learns its rank and size from Slurm.

The generated `_core.cpp` is the same for both builds (the communicator is passed to C++ as a Fortran handle, `comm.py2f()`), so switching only needs a rebuild of the extension.

The package is `pyddstore` (compiled core `pyddstore._core`, plus `pyddstore.torch`). After updating from a 1.x checkout, rebuild; an old `src/pyddstore.cpython-*.so` or `src/pyddstore.cpp` left behind is unused and can be deleted (the build warns about them).

Editable and in-place builds keep the generated `src/pyddstore/_core.cpp` in the checkout, shared by every environment that builds from it. `setup.py` regenerates it whenever the NumPy major version differs from the previous build's, because a file generated against NumPy 2 doesn't compile against NumPy 1.x headers.
