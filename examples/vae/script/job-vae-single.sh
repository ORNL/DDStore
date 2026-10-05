#!/bin/bash
#SBATCH -A FUS184
#SBATCH -J GX-single
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -N 4
#SBATCH -t 30:00
#SBATCH -q debug
#
# Baseline VAE DDP run. Runs on Frontier and Perlmutter; the rank layout is
# picked from the machine (see PLATFORM below). The #SBATCH lines above are
# for Frontier. On Perlmutter, override them on the command line:
#   sbatch -A <account> -C gpu --gpus-per-node=4 examples/vae/script/job-vae-single.sh

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
  --num-workers=N  DataLoader workers. 0 uses PyTorch's standard
                 DataLoader in the main process (no worker processes).
                 > 0 switches to ThreadDataLoader with that many worker
                 threads (requires --method=1 or 2). Default: 0.
  --replicate=R  Repeat the MNIST training set R times (longer epochs,
                 same per-sample cost). Default: 1.
  --image-scale=S  Upscale images to (28*S)x(28*S): S^2 larger samples.
                 Default: 1.
  --ranks-per-node=N  Default: 8 on Frontier, 4 on Perlmutter.
  --cpus-per-task=N   Default: 7 on Frontier, 32 on Perlmutter.
  -h, --help     Show this help message and exit.

Examples:
  $(basename "$0")                              # method=0, cxi
  $(basename "$0") --method=1 --gpudirect       # GPUDirect over libfabric
  $(basename "$0") --method=2 --gpudirect       # GPUDirect over file-based handshake
  $(basename "$0") --fabric=hsn                 # baseline over hsn instead
  $(basename "$0") --method=1 --num-workers=4   # ThreadDataLoader, no GPU buffers
  $(basename "$0") --method=1 --gpudirect --num-workers=4  # GPUDirect + ThreadDataLoader
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
NUM_WORKERS=
REPLICATE=
IMAGE_SCALE=
RANKS_PER_NODE=
CPUS_PER_TASK=
for arg in "$@"; do
    case "$arg" in
        --method=*) METHOD="${arg#--method=}" ;;
        --fabric=*) FABRIC="${arg#--fabric=}" ;;
        --gpudirect) GPUDIRECT_ARGS="--gpu-dest --gpu-source" ;;
        --num-workers=*) NUM_WORKERS="${arg#--num-workers=}" ;;
        --replicate=*) REPLICATE="${arg#--replicate=}" ;;
        --image-scale=*) IMAGE_SCALE="${arg#--image-scale=}" ;;
        --ranks-per-node=*) RANKS_PER_NODE="${arg#--ranks-per-node=}" ;;
        --cpus-per-task=*) CPUS_PER_TASK="${arg#--cpus-per-task=}" ;;
    esac
done

export DDSTORE_FABRIC="${FABRIC:-cxi}"
METHOD="${METHOD:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
REPLICATE="${REPLICATE:-1}"
IMAGE_SCALE="${IMAGE_SCALE:-1}"

EXTRA_ARGS="$GPUDIRECT_ARGS --num-workers=$NUM_WORKERS --replicate=$REPLICATE --image-scale=$IMAGE_SCALE"

echo "DDSTORE_METHOD=$METHOD DDSTORE_FABRIC=$DDSTORE_FABRIC EXTRA_ARGS=\"$EXTRA_ARGS\""

# Perlmutter: every rank sees all 4 GPUs of its node and vae-ddp.py picks
# cuda:$SLURM_LOCALID. With --gpus-per-task=1 each rank sees only its own GPU
# and NCCL (2.29, pytorch/2.13.0) fails in DDP init with "Cuda failure 101
# 'invalid device ordinal'" (transport/p2p.cc).
PLATFORM="${NERSC_HOST:-${LMOD_SYSTEM_NAME:-frontier}}"
case "$PLATFORM" in
    perlmutter)
        RANKS_PER_NODE="${RANKS_PER_NODE:-4}"
        CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
        GPU_ARGS="--gpus-per-node=4" ;;
    *)
        RANKS_PER_NODE="${RANKS_PER_NODE:-8}"
        CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
        GPU_ARGS="--gpus-per-task=1" ;;
esac
NNODES="${SLURM_NNODES:-1}"

echo "PLATFORM=$PLATFORM NNODES=$NNODES RANKS_PER_NODE=$RANKS_PER_NODE CPUS_PER_TASK=$CPUS_PER_TASK GPU_ARGS=$GPU_ARGS"

DDSTORE_METHOD=$METHOD srun -N$NNODES -n$((NNODES*RANKS_PER_NODE)) -c$CPUS_PER_TASK $GPU_ARGS -l \
    python -u examples/vae/vae-ddp.py --epochs 3 $EXTRA_ARGS \
    > >(sed 's/^/[core] /') 2> >(sed 's/^/[core] /')
