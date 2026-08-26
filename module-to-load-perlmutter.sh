#!/bin/bash

module load pytorch/2.13.0

export PYTHONNOUSERSITE=1
export MPICH_GPU_SUPPORT_ENABLED=0

## python env
source .venv/bin/activate
