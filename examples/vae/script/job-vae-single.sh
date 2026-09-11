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
  -h, --help     Show this help message and exit.

Examples:
  $(basename "$0")                              # method=0, cxi
  $(basename "$0") --method=1 --gpudirect       # GPUDirect over libfabric
  $(basename "$0") --method=2 --gpudirect       # GPUDirect over file-based handshake
  $(basename "$0") --fabric=hsn                 # baseline over hsn instead
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
EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --method=*) METHOD="${arg#--method=}" ;;
        --fabric=*) FABRIC="${arg#--fabric=}" ;;
        --gpudirect) EXTRA_ARGS="--gpu-dest --gpu-source" ;;
    esac
done

export DDSTORE_FABRIC="${FABRIC:-cxi}"
METHOD="${METHOD:-0}"

echo "DDSTORE_METHOD=$METHOD DDSTORE_FABRIC=$DDSTORE_FABRIC EXTRA_ARGS=\"$EXTRA_ARGS\""

DDSTORE_METHOD=$METHOD srun -N$SLURM_NNODES -n$((SLURM_NNODES*8)) -c7 --gpus-per-task=1 -l \
    python -u examples/vae/vae-ddp.py --epochs 3 $EXTRA_ARGS \
    > >(sed 's/^/[core] /') 2> >(sed 's/^/[core] /')
