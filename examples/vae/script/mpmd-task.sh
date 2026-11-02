#!/bin/bash
# Per-task wrapper for one MPMD srun step (srun --multi-prog) that runs the
# core and extra programs together:
#
#   mpmd-task.sh FIRST_TASK NTASKS LOCAL_OFFSET MASTER_NODE command...
#
# Slurm numbers the tasks of both programs together. This makes each program
# see only its own tasks: SLURM_PROCID counts from FIRST_TASK, SLURM_NTASKS
# is the program's size, SLURM_LOCALID counts from LOCAL_OFFSET on each node
# (the GPU the extra side picks with all of the node's GPUs visible), and
# SLURM_STEP_NODELIST names the node of the program's rank 0 (torch.distributed
# master in ddp_utils.setup_ddp).
first=$1 n=$2 local_off=$3 master=$4
shift 4
export SLURM_PROCID=$((SLURM_PROCID - first))
export SLURM_NTASKS=$n SLURM_NPROCS=$n
export SLURM_LOCALID=$((SLURM_LOCALID - local_off))
export SLURM_STEP_NODELIST=$master
exec "$@"
