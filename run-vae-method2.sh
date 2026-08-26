#!/bin/bash

rm -rf ddstore_hs_vae
mkdir -p results
sleep 2

DDSTORE_METHOD=1 MPICH_GPU_SUPPORT_ENABLED=0 MASTER_PORT=8889 srun -N$SLURM_NNODES -n$((SLURM_NNODES*4)) -c2 --gpus-per-task=0 --cpu-bind=verbose,core -l \
    python -u examples/vae/vae_core_server.py ddstore_hs_vae \
    > >(sed 's/^/[core] /') 2> >(sed 's/^/[core] /') &
sleep 5

DDSTORE_METHOD=1 MPICH_GPU_SUPPORT_ENABLED=0 MASTER_PORT=8891 DDSTORE_HANDSHAKE_TIMEOUT_S=60 srun -N$SLURM_NNODES -n$((SLURM_NNODES*4)) -c30 --gres=gpu:4 --cpu-bind=verbose,core -l \
    python -u examples/vae/vae_extra_train.py --handshake-dir ddstore_hs_vae --n-core $((SLURM_NNODES*4)) --epochs 3 \
    > >(sed 's/^/[extr] /') 2> >(sed 's/^/[extr] /')
wait
