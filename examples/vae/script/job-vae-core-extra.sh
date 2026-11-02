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
# Core/extra split VAE DDP run: two independent srun steps, or one MPMD step
# with --mpmd.
#
# Slingshot networking (Frontier): every srun step gets its own VNI, and two
# endpoints can only talk on the same VNI. --network=job_vni adds a job-wide
# VNI to every step (SLINGSHOT_VNIS=<step VNI>,<job VNI>), and single_node_vni
# makes single-node steps (split-node layout) get a CXI service at all --
# without it fi_domain() fails with -38 (ENOSYS). libfabric's cxi provider
# uses only the FIRST VNI listed, so each task below restricts
# SLINGSHOT_VNIS to the job VNI (last entry) so core and extra share it.
# With --mpmd both programs run in one step (srun --multi-prog), so they share
# the step's VNI and none of this is needed.
#
# Runs on Frontier and Perlmutter; the rank layout is picked from the machine
# (see PLATFORM below). The #SBATCH lines above are for Frontier. On
# Perlmutter, override them on the command line:
#   sbatch -A <account> -C gpu --gpus-per-node=4 examples/vae/script/job-vae-core-extra.sh

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
                 every allocated node, steps run with --overlap) or
                 split-node (core gets --core-nnodes nodes, extra the rest).
                 With two steps, colocate works on Perlmutter but not on
                 Frontier, where the second step fails with "Error
                 configuring interconnect"; use split-node or --mpmd there.
                 Default: split-node.
  --mpmd         Run core and extra in one srun step (--multi-prog, via
                 mpmd-task.sh) instead of two. Both layouts work on Frontier
                 and no job VNI is needed. Every task gets the same CPUs and
                 sees all of its node's GPUs; the extra side picks one by
                 local rank.
  --core-nnodes=N  Number of nodes for the core step in split-node layout.
                 Ignored in colocate layout. Default: 1.
  --num-workers=N  DataLoader workers for the extra (training) step:
                 threads with --loader=thread, forked processes with
                 --loader=torch, 0 = main process. Default: 0.
  --loader=X     torch: PyTorch's standard DataLoader (forked workers,
                 each with its own DDStore endpoint, host memory).
                 thread: ThreadDataLoader. Default: thread if
                 --num-workers > 0, else torch.
  --replicate=R  Repeat the MNIST training set R times on the core side
                 (longer epochs, same per-sample cost). Default: 1.
  --image-scale=S  Upscale images to (28*S)x(28*S) on the core side (the
                 extra side follows the published size). Default: 1.
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
MPMD=0
LAYOUT=
CORE_NNODES_OPT=
NUM_WORKERS=
LOADER=
REPLICATE=
IMAGE_SCALE=
for arg in "$@"; do
    case "$arg" in
        --method=*) METHOD="${arg#--method=}" ;;
        --fabric=*) FABRIC="${arg#--fabric=}" ;;
        --gpudirect) GPUDIRECT=1 ;;
        --mpmd) MPMD=1 ;;
        --layout=*) LAYOUT="${arg#--layout=}" ;;
        --core-nnodes=*) CORE_NNODES_OPT="${arg#--core-nnodes=}" ;;
        --num-workers=*) NUM_WORKERS="${arg#--num-workers=}" ;;
        --loader=*) LOADER="${arg#--loader=}" ;;
        --replicate=*) REPLICATE="${arg#--replicate=}" ;;
        --image-scale=*) IMAGE_SCALE="${arg#--image-scale=}" ;;
    esac
done
LAYOUT="${LAYOUT:-split-node}"
NUM_WORKERS="${NUM_WORKERS:-0}"
REPLICATE="${REPLICATE:-1}"
IMAGE_SCALE="${IMAGE_SCALE:-1}"

export DDSTORE_FABRIC="${FABRIC:-cxi}"
# CPU->NIC map, computed once here instead of a topology query in every rank
# (docs/backends.md). Empty on failure: ranks then query the topology themselves.
export DDSTORE_NIC_MAP="${DDSTORE_NIC_MAP:-$(python3 -m cpu_nic_map --env)}"
export DDSTORE_METHOD="${METHOD:-2}"
export VAE_PROFILE=1
# Forked DataLoader workers read DDStore through their own endpoints; that
# needs a fork-safe libfabric MR cache monitor (see docs/pytorch.md).
if [ "$LOADER" = torch ] && [ "$NUM_WORKERS" -gt 0 ]; then
    export DDSTORE_FORK_WORKERS=1
fi

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
    # Both steps share the nodes; Perlmutter needs --overlap for that.
    OVERLAP=--overlap
else
    CORE_NNODES="${CORE_NNODES_OPT:-1}"
    EXTRA_NNODES=$((SLURM_NNODES - CORE_NNODES))
fi
# Perlmutter: the extra (training) ranks see all 4 GPUs of their node and
# vae_extra_train.py picks cuda:$SLURM_LOCALID. With --gpus-per-task=1 NCCL
# (2.29, pytorch/2.13.0) fails in DDP init with "Cuda failure 101 'invalid
# device ordinal'". The core step does no NCCL, so --gpus-per-task is fine
# there; 4 core ranks per node fit the 4 GPUs with --gpudirect.
PLATFORM="${NERSC_HOST:-${LMOD_SYSTEM_NAME:-frontier}}"
case "$PLATFORM" in
    perlmutter)
        CORE_NR=4
        EXTRA_NR=4
        EXTRA_CPUS_PER_TASK=16
        GPUS_PER_NODE=4
        EXTRA_GPU_ARGS="--gpus-per-node=4" ;;
    *)
        CORE_NR=8
        EXTRA_NR=8
        EXTRA_CPUS_PER_TASK=6
        GPUS_PER_NODE=8
        EXTRA_GPU_ARGS="--gpus-per-task=1" ;;
esac
CORE_NTASKS=$((CORE_NNODES * CORE_NR))
EXTRA_NTASKS=$((EXTRA_NNODES * EXTRA_NR))

echo "PLATFORM=$PLATFORM DDSTORE_METHOD=$DDSTORE_METHOD DDSTORE_FABRIC=$DDSTORE_FABRIC LAYOUT=$LAYOUT MPMD=$MPMD GPUDIRECT=$GPUDIRECT"
echo "CORE_NNODES=$CORE_NNODES CORE_NTASKS=$CORE_NTASKS CORE_GPUS_PER_TASK=$CORE_GPUS_PER_TASK CORE_EXTRA_ARGS=\"$CORE_EXTRA_ARGS\" REPLICATE=$REPLICATE IMAGE_SCALE=$IMAGE_SCALE"
echo "EXTRA_NNODES=$EXTRA_NNODES EXTRA_NTASKS=$EXTRA_NTASKS EXTRA_EXTRA_ARGS=\"$EXTRA_EXTRA_ARGS\" NUM_WORKERS=$NUM_WORKERS"

CORE_CMD="python -u examples/vae/vae_core_server.py ddstore_hs_vae --replicate=$REPLICATE --image-scale=$IMAGE_SCALE $CORE_EXTRA_ARGS"
EXTRA_CMD="python -u examples/vae/vae_extra_train.py --handshake-dir ddstore_hs_vae --n-core $CORE_NTASKS --epochs 3 --num-workers=$NUM_WORKERS${LOADER:+ --loader=$LOADER} $EXTRA_EXTRA_ARGS"

if [ "$MPMD" == "1" ]; then
    # One step: tasks 0..CORE_NTASKS-1 run the core, the rest the extra side.
    # colocate: cyclic puts CORE_NR core tasks (local ranks 0..) and then
    # EXTRA_NR extra tasks on every node, so the extra local rank starts at
    # CORE_NR. split-node: block fills the first CORE_NNODES nodes with core
    # tasks. The core exchanges its records through files, not MPI, since
    # MPI_COMM_WORLD would span both programs.
    mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    if [ "$LAYOUT" == "colocate" ]; then
        DIST=cyclic
        EXTRA_LOCAL0=$CORE_NR
        EXTRA_MASTER=${HOSTS[0]}
        CPUS_PER_TASK=$((EXTRA_CPUS_PER_TASK / 2))
    else
        DIST=block
        EXTRA_LOCAL0=0
        EXTRA_MASTER=${HOSTS[$CORE_NNODES]}
        CPUS_PER_TASK=$EXTRA_CPUS_PER_TASK
    fi
    CONF=results/mpmd-${SLURM_JOB_ID}.conf
    TASK=examples/vae/script/mpmd-task.sh
    cat > $CONF <<EOF
0-$((CORE_NTASKS - 1)) env DDSTORE_COMM=file $TASK 0 $CORE_NTASKS 0 ${HOSTS[0]} $CORE_CMD
$CORE_NTASKS-$((CORE_NTASKS + EXTRA_NTASKS - 1)) env DDSTORE_HANDSHAKE_TIMEOUT_S=60 $TASK $CORE_NTASKS $EXTRA_NTASKS $EXTRA_LOCAL0 $EXTRA_MASTER $EXTRA_CMD
EOF
    cat $CONF
    MASTER_PORT=8891 srun -N$SLURM_NNODES -n$((CORE_NTASKS + EXTRA_NTASKS)) -c$CPUS_PER_TASK --gpus-per-node=$GPUS_PER_NODE \
        -m $DIST --cpu-bind=verbose,core -l --multi-prog $CONF
    exit
fi

MASTER_PORT=8889 srun ${OVERLAP:-} -N$CORE_NNODES -n$CORE_NTASKS -c1 --gpus-per-task=$CORE_GPUS_PER_TASK --cpu-bind=verbose,core -l \
    bash -c "$JOB_VNI_WRAP" _ $CORE_CMD \
    > >(sed 's/^/[core] /') 2> >(sed 's/^/[core] /') &
sleep 5

MASTER_PORT=8891 DDSTORE_HANDSHAKE_TIMEOUT_S=60 srun ${OVERLAP:-} -N$EXTRA_NNODES -n$EXTRA_NTASKS -c$EXTRA_CPUS_PER_TASK $EXTRA_GPU_ARGS --cpu-bind=verbose,core -l \
    bash -c "$JOB_VNI_WRAP" _ $EXTRA_CMD \
    > >(sed 's/^/[extr] /') 2> >(sed 's/^/[extr] /')
sleep 5

wait
