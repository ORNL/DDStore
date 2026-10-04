#!/bin/bash
#SBATCH -A FUS184
#SBATCH -J GX-core-extra
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -N 4
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH --network=single_node_vni,job_vni
#
# Core/extra split VAE DDP run (two independent srun steps).
#
# Slingshot networking (Frontier): every srun step gets its own VNI, and two
# endpoints can only talk on the same VNI. --network=job_vni adds a job-wide
# VNI to every step (SLINGSHOT_VNIS=<step VNI>,<job VNI>), and single_node_vni
# makes single-node steps (split-node layout) get a CXI service at all --
# without it fi_domain() fails with -38 (ENOSYS). libfabric's cxi provider
# uses only the FIRST VNI listed, so each task below restricts
# SLINGSHOT_VNIS to the job VNI (last entry) so core and extra share it.

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Runs the core/extra VAE DDP split (default method=2).

Options:
  --method=N     Exported as DDSTORE_METHOD, but the core/extra split is
                 always method=2 (file-based handshake): vae_core_server.py
                 sets it and vae_extra_train.py always joins via method 2.
                 Default: 2.
  --fabric=X     DDSTORE_FABRIC: hsn or cxi. Default: cxi.
  --gpudirect    Test GPUDirect RDMA. Requires --method=1 or 2 and
                 --fabric=cxi. Also gives the core step a GPU per rank
                 (needed for --gpu-source; the baseline core step doesn't
                 use one).
  --layout=X     Process distribution: colocate (core and extra both span
                 every allocated node) or split-node (core gets 1 node,
                 extra gets the rest). Default: colocate.
  --core-nnodes=N  Number of nodes for the core step in split-node layout.
                 Ignored in colocate layout. Default: 1.
  --num-workers=N  DataLoader workers for the extra (training) step. 0 uses
                 PyTorch's standard DataLoader in the main process. > 0
                 switches to ThreadDataLoader with that many worker threads.
                 Default: 0.
  -h, --help     Show this help message and exit.
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
    esac
done

METHOD=
FABRIC=
GPUDIRECT=0
LAYOUT=
CORE_NNODES_OPT=
NUM_WORKERS=
for arg in "$@"; do
    case "$arg" in
        --method=*) METHOD="${arg#--method=}" ;;
        --fabric=*) FABRIC="${arg#--fabric=}" ;;
        --gpudirect) GPUDIRECT=1 ;;
        --layout=*) LAYOUT="${arg#--layout=}" ;;
        --core-nnodes=*) CORE_NNODES_OPT="${arg#--core-nnodes=}" ;;
        --num-workers=*) NUM_WORKERS="${arg#--num-workers=}" ;;
    esac
done
LAYOUT="${LAYOUT:-colocate}"
NUM_WORKERS="${NUM_WORKERS:-0}"

export DDSTORE_FABRIC="${FABRIC:-cxi}"
export DDSTORE_METHOD="${METHOD:-2}"

if [ "$GPUDIRECT" == "1" ]; then
    CORE_GPUS_PER_TASK=1
    CORE_EXTRA_ARGS="--gpu-source"
    EXTRA_EXTRA_ARGS="--gpu-dest"
else
    CORE_GPUS_PER_TASK=0
    CORE_EXTRA_ARGS=""
    EXTRA_EXTRA_ARGS=""
fi

# Per-task wrapper: use only the job VNI (see the header comment).
JOB_VNI_WRAP='export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}; exec "$@"'

rm -rf ddstore_hs_vae
mkdir -p results
sleep 2

if [ "$LAYOUT" == "colocate" ]; then
    CORE_NNODES=$SLURM_NNODES
    EXTRA_NNODES=$SLURM_NNODES
else
    CORE_NNODES="${CORE_NNODES_OPT:-1}"
    EXTRA_NNODES=$((SLURM_NNODES - CORE_NNODES))
fi
CORE_NR=8
EXTRA_NR=8
CORE_NTASKS=$((CORE_NNODES * CORE_NR))
EXTRA_NTASKS=$((EXTRA_NNODES * EXTRA_NR))

echo "DDSTORE_METHOD=$DDSTORE_METHOD DDSTORE_FABRIC=$DDSTORE_FABRIC LAYOUT=$LAYOUT GPUDIRECT=$GPUDIRECT"
echo "CORE_NNODES=$CORE_NNODES CORE_NTASKS=$CORE_NTASKS CORE_GPUS_PER_TASK=$CORE_GPUS_PER_TASK CORE_EXTRA_ARGS=\"$CORE_EXTRA_ARGS\""
echo "EXTRA_NNODES=$EXTRA_NNODES EXTRA_NTASKS=$EXTRA_NTASKS EXTRA_EXTRA_ARGS=\"$EXTRA_EXTRA_ARGS\" NUM_WORKERS=$NUM_WORKERS"

MASTER_PORT=8889 srun -N$CORE_NNODES -n$CORE_NTASKS -c1 --gpus-per-task=$CORE_GPUS_PER_TASK --cpu-bind=verbose,core -l \
    bash -c "$JOB_VNI_WRAP" _ python -u examples/vae/vae_core_server.py ddstore_hs_vae $CORE_EXTRA_ARGS \
    > >(sed 's/^/[core] /') 2> >(sed 's/^/[core] /') &
sleep 5

MASTER_PORT=8891 DDSTORE_HANDSHAKE_TIMEOUT_S=60 srun -N$EXTRA_NNODES -n$EXTRA_NTASKS -c6 --gpus-per-task=1 --cpu-bind=verbose,core -l \
    bash -c "$JOB_VNI_WRAP" _ python -u examples/vae/vae_extra_train.py --handshake-dir ddstore_hs_vae --n-core $CORE_NTASKS --epochs 3 --num-workers=$NUM_WORKERS $EXTRA_EXTRA_ARGS \
    > >(sed 's/^/[extr] /') 2> >(sed 's/^/[extr] /')
sleep 5

wait
