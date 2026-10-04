# Perlmutter checklist for `check-thread`

Everything on this branch was verified on Frontier (AMD MI250X, ROCm, cxi).
These are the things Frontier could not cover, and how to check them on
Perlmutter (NVIDIA A100, CUDA, cxi, 4 GPUs and 64 cores per node).

## Run it

From the repo root on Perlmutter, with your environment loaded (modules, a
venv with CUDA torch + mpi4py) and the branch built:

```bash
git fetch origin && git checkout check-thread && git pull
CC=cc CXX=CC pip install --no-build-isolation --no-deps -e .
sbatch -A <account> examples/scripts/perlmutter-check.sh
```

About 15–25 min on 2 debug nodes. Send back `pm-check-<jobid>/summary.txt`
(logs are in the same directory).

If the build fails with `No module named 'distutils.msvccompiler'`, prefix it
with `SETUPTOOLS_USE_DISTUTILS=stdlib`.

## What it checks, and what "good" looks like

| # | Check | Why it matters | Expect |
|---|---|---|---|
| 1 | Slurm switch config; `SLINGSHOT_VNIS` per step | The core/extra fix keeps the **last** VNI (job VNI) because cxi uses the first. Perlmutter's order must match. | Each step: `<own VNI>,<job VNI>`, job VNI identical in both, last |
| 2 | NIC picked per rank (`cpu_nic_map`) | 4 NICs per node, `hsnN` → `cxiN` | Ranks on different NICs, names `cxi0..cxi3` |
| 3 | `test_single`, `test_multirank` | method 0 basics | all pass |
| 4 | `test_get_batch` on 8 ranks | batched get, method 0 collective + method 1 cxi, threads | all pass (the GPU case runs here) |
| 5 | `test_gpu_rdma` on 2 ranks | **CUDA GPUDirect (`FI_HMEM_CUDA`) was never run on this branch** | all pass |
| 6 | VAE, method 1, host/GPU, 0 and 2 workers, batched on/off | batched vs per-sample on CUDA | rc=0, 0 errors, **same loss8 for B0 and B1** within each pair |
| 7 | VAE, method 0 per-row vs collective | MDLoader-style collective | same loss8 for B0 and B1 |
| 8 | core/extra **split-node**, host and `--gpu-dest` | VNI wrapper + single-node steps (`single_node_vni`) | extra epoch3 printed, `core_done=4/4`, 0 errors |
| 9 | core/extra **colocate** | fails on Frontier with `job_vni` (`Error configuring interconnect`) | your test: does it launch and train? |
| 10 | `bench_get.py` | sanity of batched speedup on A100 | batch 128 much faster per row than batch 1 |

Notes:
- If `sbatch` rejects `--network=single_node_vni,job_vni`, delete that line
  from the script and note it: the split-node checks then show whether
  Perlmutter needs it.
- Losses on A100 will differ from Frontier's (8.8960 / 30.0178); compare
  batched vs per-sample on Perlmutter itself.
- The repo's `job-vae-single.sh` / `job-vae-core-extra.sh` assume Frontier
  (8 ranks × 7 cores per node, `-A FUS184`); this check script uses `srun`
  directly with 4 ranks per node (`RANKS_PER_NODE`, `CPUS_PER_TASK`).
- If colocate fails the same way as on Frontier, the README's limitation
  applies to both machines; if it works, note which `SwitchParameters`
  Perlmutter uses.
