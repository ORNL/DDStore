#!/bin/bash

module load PrgEnv-gnu
module load cpe/24.11
module load libfabric
module unload darshan-runtime
module load cray-python/3.11.7
module load rocm/7.2.0

export PYTHONNOUSERSITE=1
export DDSTORE_FABRIC_PROVIDER=hsn

## python env
source .venv/bin/activate

