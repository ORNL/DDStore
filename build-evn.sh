#!/bin/bash

# Load system modules
module load pytorch/2.13.0
module unload darshan

export PYTHONNOUSERSITE=1
export MPICH_GPU_SUPPORT_ENABLED=0
export DDSTORE_FABRIC_PROVIDER=cxi

VENV_DIR=.venv
python -m venv --system-site-packages "${VENV_DIR}"

source .venv/bin/activate
pip install --upgrade pip

# Install build and runtime dependencies
pip install wheel Cython
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py==4.1.1
pip install pytest pytest-mpi
pip install psutil

# Build and install PyDDStore using Cray compiler wrappers
CC=cc CXX=CC pip install -e .

# Install PyTorch with ROCm support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

echo "Environment is ready at '.venv'."
echo "To activate: source .venv/bin/activate"
