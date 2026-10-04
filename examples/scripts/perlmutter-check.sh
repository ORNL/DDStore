#!/bin/bash
#SBATCH -J ddstore-pm-check
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 30:00
#SBATCH --gpus-per-node=4
#SBATCH --network=single_node_vni,job_vni
#
# Correctness check of the check-thread branch on Perlmutter (NERSC).
#
# Usage, from the repo root with your Perlmutter Python env loaded (modules,
# venv with torch + mpi4py, pyddstore built with
#   CC=cc CXX=CC pip install --no-build-isolation --no-deps -e . ):
#   sbatch -A <account> examples/scripts/perlmutter-check.sh
# Results: pm-check-<jobid>/summary.txt (send that back), logs alongside.
#
# Knobs: RANKS_PER_NODE (4), CPUS_PER_TASK (32).

set -u
cd "$SLURM_SUBMIT_DIR"
NR=${RANKS_PER_NODE:-4}
CPT=${CPUS_PER_TASK:-32}
N=$SLURM_NNODES
NT=$((N * NR))
OUT=pm-check-$SLURM_JOB_ID
mkdir -p "$OUT"
SUM=$OUT/summary.txt
export DDSTORE_FABRIC=cxi
SRUN="srun -N$N -n$NT -c$CPT --gpus-per-task=1"
NOISE='Tip:|DDSTORE_NIC_MAP=|using interface|endpoint max_msg|DDStore\]|Step created'

say() { echo "$*" | tee -a "$SUM"; }
section() { say ""; say "=== $*"; }

# ---------------------------------------------------------------- system
section "system"
scontrol show config | grep -iE "SwitchType|SwitchParameters" | tee -a "$SUM"
python - <<'EOF' 2>&1 | tee -a "$SUM"
import torch, numpy, mpi4py
print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", torch.version.hip,
      "gpus/node", torch.cuda.device_count(), "| numpy", numpy.__version__, "| mpi4py", mpi4py.__version__)
EOF
(fi_info --version 2>/dev/null | head -2) | tee -a "$SUM"
say "-- per-step Slingshot env (each step should show its own VNI, then the job VNI, last)"
for r in 0 1; do
  srun -N1 -n1 --relative=$r bash -c 'echo "  step $SLURM_STEP_ID node $(hostname): SLINGSHOT_VNIS=$SLINGSHOT_VNIS SVC_IDS=$SLINGSHOT_SVC_IDS DEVICES=$SLINGSHOT_DEVICES"' 2>&1 | tee -a "$SUM"
done
say "-- NIC picked per rank (cpu_nic_map, cxi)"
$SRUN python -c "
import os, cpu_nic_map as m
iface = m.select_fabric_iface()
print('  rank', os.environ.get('SLURM_PROCID'), 'cpus', sorted(os.sched_getaffinity(0))[:3], '... ->', iface)
" 2>&1 | grep -vE "$NOISE" | sort -k2 -n | tee -a "$SUM"

# ---------------------------------------------------------------- tests
section "unit tests"
srun -N1 -n1 -c$CPT --gpus-per-task=1 python -m pytest -q -p no:cacheprovider test/test_single.py > $OUT/t_single.log 2>&1
say "test_single (1 rank):          $(tail -1 $OUT/t_single.log)"
srun -N1 -n$NR -c$CPT --gpus-per-task=1 python -m pytest -q -p no:cacheprovider test/test_multirank.py > $OUT/t_multirank.log 2>&1
say "test_multirank ($NR ranks):      $(grep -E 'passed|failed' $OUT/t_multirank.log | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
$SRUN python -m pytest -q -p no:cacheprovider -rs test/test_get_batch.py > $OUT/t_get_batch.log 2>&1
say "test_get_batch ($NT ranks, cxi): $(grep -E 'passed|failed' $OUT/t_get_batch.log | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
srun -N2 -n2 -c$CPT --gpus-per-task=1 python -m pytest -q -p no:cacheprovider test/test_gpu_rdma.py > $OUT/t_gpu_rdma.log 2>&1
say "test_gpu_rdma (2 ranks, CUDA): $(grep -E 'passed|failed' $OUT/t_gpu_rdma.log | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
grep -E "^FAILED|Error" $OUT/t_*.log | sort | uniq | head -20 | tee -a "$SUM"

# ---------------------------------------------------------------- VAE
section "vae-ddp ($NT ranks, 8 epochs; batched vs per-sample must give the same loss)"
export VAE_PROFILE=1
vae() {  # tag, env..., -- args...
  local tag=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -rf ddstore_hs*
  env "${envs[@]}" $SRUN -l python -u examples/vae/vae-ddp.py --epochs 8 "$@" > $OUT/vae-$tag.log 2>&1
  local rc=$?
  local loss=$(grep -h ' 0: ====> Epoch: 8 Average' $OUT/vae-$tag.log | awk '{print $NF}')
  local ep=$(grep -h '\[profile\]' $OUT/vae-$tag.log | awk '{f=$5;c=$6;sub("fetch=","",f);sub("s","",f);sub("compute=","",c);sub("s","",c);F+=f;C+=c;n++} END{if(n)printf "fetch=%.3f compute=%.3f total=%.3f",F/n,C/n,(F+C)/n}')
  say "$(printf '%-22s' $tag) rc=$rc loss8=$loss $ep errors=$(grep -ciE 'traceback|error|abort' $OUT/vae-$tag.log)"
}
for B in 0 1; do
  vae m1_host_w0_B$B DDSTORE_METHOD=1 DDSTORE_BATCH_GET=$B -- --num-workers=0
  vae m1_host_w2_B$B DDSTORE_METHOD=1 DDSTORE_BATCH_GET=$B -- --num-workers=2
  vae m1_gpu_w0_B$B  DDSTORE_METHOD=1 DDSTORE_BATCH_GET=$B -- --num-workers=0 --gpu-dest --gpu-source
  vae m1_gpu_w2_B$B  DDSTORE_METHOD=1 DDSTORE_BATCH_GET=$B -- --num-workers=2 --gpu-dest --gpu-source
  vae m0_host_w0_B$B DDSTORE_METHOD=0 DDSTORE_BATCH_GET=$B -- --num-workers=0
done
vae m1_gpu_w0_B1_S2 DDSTORE_METHOD=1 DDSTORE_BATCH_GET=1 -- --num-workers=0 --gpu-dest --gpu-source --image-scale=2
vae m1_host_w0_B1_S2 DDSTORE_METHOD=1 DDSTORE_BATCH_GET=1 -- --num-workers=0 --image-scale=2

# ---------------------------------------------------------------- core/extra
section "method 2 core/extra (separate srun steps, job VNI)"
# libfabric cxi uses the FIRST VNI in SLINGSHOT_VNIS; keep only the job VNI
# (last entry) so both steps share it. See README "Multiple srun steps".
WRAP='export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}; exec "$@"'
corextra() {  # tag, core srun opts, extra srun opts, extra-args
  local tag=$1 copts=$2 eopts=$3 eargs=$4 hs=ddstore_hs_pm_$1
  rm -rf "$hs"
  DDSTORE_METHOD=2 DDSTORE_HANDSHAKE_TIMEOUT_S=120 MASTER_PORT=8889 timeout 600 srun $copts -l \
      bash -c "$WRAP" _ python -u examples/vae/vae_core_server.py "$hs" > $OUT/ce-$tag-core.log 2>&1 &
  local cpid=$!
  sleep 5
  DDSTORE_METHOD=2 DDSTORE_HANDSHAKE_TIMEOUT_S=120 MASTER_PORT=8891 timeout 600 srun $eopts -l \
      bash -c "$WRAP" _ python -u examples/vae/vae_extra_train.py --handshake-dir "$hs" --n-core $NR --epochs 3 $eargs \
      > $OUT/ce-$tag-extra.log 2>&1
  local erc=$?
  wait $cpid; local crc=$?
  say "$(printf '%-22s' $tag) extra rc=$erc epoch3=$(grep -h ' 0: ====> Epoch: 3 Average' $OUT/ce-$tag-extra.log | awk '{print $NF}') core rc=$crc core_done=$(grep -c 'core rank [0-9]*\] done' $OUT/ce-$tag-core.log)/$NR errors=$(cat $OUT/ce-$tag-*.log | grep -ciE 'traceback|error|abort|VNI_NOT|PTLTE|interconnect')"
  grep -hoE "Error configuring interconnect|VNI_NOT_FOUND|fi_domain\(\) has failed" $OUT/ce-$tag-*.log | sort | uniq -c | sed 's/^/    /' | tee -a "$SUM"
}
corextra split_host   "-N1 -n$NR --relative=0 -c$CPT --gpus-per-task=0" "-N1 -n$NR --relative=1 -c$CPT --gpus-per-task=1" "--num-workers=1"
corextra split_gpudest "-N1 -n$NR --relative=0 -c$CPT --gpus-per-task=0" "-N1 -n$NR --relative=1 -c$CPT --gpus-per-task=1" "--num-workers=1 --gpu-dest"
# colocate: both steps on both nodes at once (fails on Frontier with job_vni).
# Core ranks split as NR/2 per node so --n-core still equals NR.
corextra colocate_host "-N2 -n$NR -c8 --gpus-per-task=0" "-N2 -n$NR -c16 --gpus-per-task=1" "--num-workers=1"

# ---------------------------------------------------------------- bench
section "bench_get.py (method 1, host/GPU, 1 row vs 128 rows per call; us/row)"
DDSTORE_PROFILE=1 DDSTORE_METHOD=1 $SRUN python -u examples/scripts/bench_get.py \
    --row-floats 784,3136,262144 --dest host,gpu --batch 1,128 --nget 2048 2>&1 | grep -vE "$NOISE" | tee -a "$SUM"

section "done"
say "Logs: $OUT/"
