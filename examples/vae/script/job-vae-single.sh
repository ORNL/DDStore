#!/bin/bash
#SBATCH -A FUS184
#SBATCH -J GX-single
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -N 4
#SBATCH -t 30:00
#SBATCH -q debug
#
# Baseline VAE DDP run.

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Runs examples/vae/vae-ddp.py under DDP.

Options:
  --method=N     DDSTORE_METHOD: 0=MPI RMA, 1=libfabric, 2=file-based
                 handshake. Default: 0.
  --fabric=X     DDSTORE_FABRIC: hsn or cxi. Default: cxi.
  --gpudirect    Test GPUDirect RDMA. Requires --method=1 or 2 and
                 --fabric=cxi.
  --thread       Use ThreadDataLoader (thread-pool DataLoader) instead of the
                 default forked-process DataLoader. Needed for
                 --num-workers > 1. Requires --method=1 or 2.
  --num-workers=N  Worker processes for the default loader (must stay <= 1
                 -- forking after MPI_Init hangs with DDStore for N > 1;
                 use --thread instead), or worker threads with --thread.
                 With --gpudirect, must stay <= 1 even with --thread --
                 confirmed unsafe above that (silent data corruption, not
                 a crash; see README Known Limitations). Default: 1.
  -h, --help     Show this help message and exit.

Examples:
  $(basename "$0")                              # method=0, cxi
  $(basename "$0") --method=1 --gpudirect       # GPUDirect over libfabric
  $(basename "$0") --method=2 --gpudirect       # GPUDirect over file-based handshake
  $(basename "$0") --fabric=hsn                 # baseline over hsn instead
  $(basename "$0") --method=1 --thread --num-workers=4        # threaded loader, no GPU buffers
  $(basename "$0") --method=1 --gpudirect --thread --num-workers=1  # GPUDirect, threaded loader
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
    esac
done

rm -rf ddstore_hs*
mkdir -p results
sleep 2

export VAE_PROFILE=1

METHOD=
FABRIC=
GPUDIRECT_ARGS=""
THREAD=0
NUM_WORKERS=
for arg in "$@"; do
    case "$arg" in
        --method=*) METHOD="${arg#--method=}" ;;
        --fabric=*) FABRIC="${arg#--fabric=}" ;;
        --gpudirect) GPUDIRECT_ARGS="--gpu-dest --gpu-source" ;;
        --thread) THREAD=1 ;;
        --num-workers=*) NUM_WORKERS="${arg#--num-workers=}" ;;
    esac
done

export DDSTORE_FABRIC="${FABRIC:-cxi}"
METHOD="${METHOD:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

EXTRA_ARGS="$GPUDIRECT_ARGS --num-workers=$NUM_WORKERS"
if [ "$THREAD" == "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --loader=threaded"
fi

echo "DDSTORE_METHOD=$METHOD DDSTORE_FABRIC=$DDSTORE_FABRIC EXTRA_ARGS=\"$EXTRA_ARGS\""

DDSTORE_METHOD=$METHOD srun -N$SLURM_NNODES -n$((SLURM_NNODES*8)) -c7 --gpus-per-task=1 -l \
    python -u examples/vae/vae-ddp.py --epochs 3 $EXTRA_ARGS \
    > >(sed 's/^/[core] /') 2> >(sed 's/^/[core] /')
