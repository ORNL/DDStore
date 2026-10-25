# Perlmutter check (branch `mpi-optional`)

Validated on Frontier only. Questions for Perlmutter (CUDA, `cxi`):

1. Forked `DataLoader` workers: do they work, and is `FI_MR_CACHE_MONITOR=memhooks`
   (set by `DDSTORE_FORK_WORKERS=1`) needed? Is `FI_CXI_DISABLE_HOST_REGISTER=1` needed?
2. `ThreadDataLoader` per-thread readers (`DDSTORE_THREAD_READERS`) and direct batch
   reads (`DDSTORE_FAST_COLLATE`): correct, and any speed difference?
3. Both builds (MPI and `DDSTORE_MPI=0`) pass the tests.

Run everything inside a 2-node GPU allocation, e.g.
`salloc -N2 -C gpu -q interactive -t 1:00:00 -A <account> --gpus-per-node=4`.

## 0. Environment

```bash
echo "FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-unset}"; env | grep ^FI_
srun -N1 -n1 fi_info --version
export DDSTORE_FABRIC=cxi DDSTORE_HANDSHAKE_TIMEOUT_S=60
```

## 1. No-MPI build

```bash
git fetch origin && git checkout mpi-optional && git pull
DDSTORE_MPI=0 CC=gcc CXX=g++ pip install --no-build-isolation --no-deps -e .
python -c "import pyddstore; print('HAVE_MPI', pyddstore.HAVE_MPI)"     # False
```

Add `SETUPTOOLS_USE_DISTUTILS=stdlib` if the build fails with `No module named 'distutils.msvccompiler'`.

```bash
S="srun -N2 -n4 --gpus-per-task=1"; T="python -m pytest -q -rf"
$S $T test/test_torch_nompi.py                                   # a. 14 passed
DDSTORE_FORK_WORKERS=0 $S $T test/test_torch_nompi.py            # b. site monitor after fork
FI_CXI_DISABLE_HOST_REGISTER=0 $S $T test/test_torch_nompi.py -k fork   # c. host register after fork
$S $T test/test_file_exchange.py                                 # 12 passed, 1 skipped
```

(b) fails at once with "needs DDSTORE_FORK_WORKERS=1" under `kdreg2`; with an unset
monitor libfabric uses its default. (c) failed on Frontier (`fi_enable` -262, ROCm).

## 2. MPI build

```bash
CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .               # or mpicc/mpicxx
python -c "import pyddstore; print('HAVE_MPI', pyddstore.HAVE_MPI)"     # True
$S $T test/test_torch.py test/test_get_batch.py                  # 126 passed
$S $T test/test_gpu_rdma.py                    # 14 passed
$S $T test/test_torch_nompi.py                                   # 14 passed (run alone)
```

## 3. VAE (MPI build, same allocation)

Losses must match within each group; compare `[profile] epoch 3` (fetch / compute).

```bash
L="grep -E 'train_loader|Epoch: 3 Average|profile\] epoch 3|Error'"
J=examples/vae/script/job-vae-single.sh
bash $J --method=1 2>&1 | eval $L                                          # baseline, 0 workers
bash $J --method=1 --loader=torch --num-workers=4 2>&1 | eval $L           # forked, persistent
bash $J --method=1 --num-workers=2 2>&1 | eval $L                          # threads, own readers
DDSTORE_THREAD_READERS=0 bash $J --method=1 --num-workers=2 2>&1 | eval $L # threads, shared store
DDSTORE_FAST_COLLATE=0 bash $J --method=1 --num-workers=2 2>&1 | eval $L   # per-sample collate
bash $J --method=1 --gpudirect --num-workers=2 2>&1 | eval $L              # GPU destination
bash examples/vae/script/job-vae-core-extra.sh --loader=torch --num-workers=2 2>&1 | eval $L
bash examples/vae/script/job-vae-core-extra.sh --loader=thread --num-workers=2 2>&1 | eval $L
```

Optional: `--image-scale=4` on the thread runs, and `DDSTORE_PROFILE=1` for the
`[ddstore-profile] per call` line (lock wait, other).

## Send back

- Step 0 output; pass/fail lines of 1a–1c, 2; VAE losses and `[profile] epoch 3` lines.
- On a hang: rerun that case with `FI_LOG_LEVEL=warn`, send the `warn>` lines.
