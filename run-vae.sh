#!/bin/bash

# method=2's file-based handshake can't tell a leftover rank file from a
# previous run apart from a fresh one (it just polls for the right file
# size), so a stale file here can hand out a dead fabric address/key.
rm -rf "${DDSTORE_HANDSHAKE_DIR:-ddstore_hs}"
mkdir -p results
sleep 2

DDSTORE_METHOD=2 srun -N$SLURM_NNODES -n$((SLURM_NNODES*4)) -c32 --gres=gpu:4 -l \
    python -u examples/vae/vae-ddp.py --epochs 3 \
    > >(tee -a run.log | sed 's/^/[core] /') 2> >(tee -a run.log | sed 's/^/[core] /')
