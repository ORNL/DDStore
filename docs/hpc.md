# HPC systems (Slurm, Slingshot)

## Slurm job scripts

[job-vae-single.sh](https://github.com/ORNL/DDStore/blob/main/examples/vae/script/job-vae-single.sh) runs `vae-ddp.py` as one `srun` step; [job-vae-core-extra.sh](https://github.com/ORNL/DDStore/blob/main/examples/vae/script/job-vae-core-extra.sh) runs the core/extra split as two steps. `--help` lists the options. The `#SBATCH` lines target Frontier (8 ranks × 7 cores per node).

```bash
sbatch examples/vae/script/job-vae-single.sh --method=1 --num-workers=1
sbatch examples/vae/script/job-vae-single.sh --method=1 --gpudirect --image-scale=2
sbatch examples/vae/script/job-vae-core-extra.sh                       # split-node: 1 core node, the rest extra
sbatch examples/vae/script/job-vae-core-extra.sh --gpudirect --core-nnodes=2
```

For your own scripts using methods 1/2, export the CPU→NIC map in the batch step before `srun` so ranks skip the per-process `hwloc` query ([details](backends.md#libfabric-rdma-method1)):

```bash
export DDSTORE_NIC_MAP=$(python3 -m cpu_nic_map --env)
```

On Perlmutter (detected via `NERSC_HOST`) the scripts use 4 ranks per node, each seeing its node's 4 GPUs (needed by NCCL). Override the `#SBATCH` lines:

```bash
sbatch -A <account> -C gpu --gpus-per-node=4 examples/vae/script/job-vae-single.sh --method=1
```

## Slingshot VNIs and `--network`

Each `srun` step gets its own VNI (network isolation ID); endpoints reach each other only on a shared VNI.

| Layout | Perlmutter, no flags | Perlmutter, `single_node_vni,job_vni` | Frontier |
|---|---|---|---|
| One `srun` step (normal training) | works | works | works; one-node steps need `single_node_vni` |
| Core and extra as separate one-node steps (`--layout=split-node`) | works | needs `job_vni` + wrapper | needs both flags + wrapper |
| Core and extra on the same nodes, or multi-node steps | extra can't reach core | needs `job_vni` + wrapper + `srun --overlap` | second step fails to launch |

`job-vae-core-extra.sh` does both parts for its two steps:

1. `#SBATCH --network=single_node_vni,job_vni`: `job_vni` adds a job-wide VNI to every step (`SLINGSHOT_VNIS=<step>,<job>`); on Frontier `single_node_vni` also gives one-node steps a CXI service.
2. The wrapper, in each task before Python starts: `export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}`. The cxi provider uses only the first VNI, so without it reads fail with `VNI_NOT_FOUND`.

It defaults to `--layout=split-node`; `--layout=colocate` works on Perlmutter only. Perlmutter columns measured on 2 nodes ([results](results.md#perlmutter-validation-2-nodes--4-a100-cuda-13-cxi)).

## Troubleshooting (`cxi`)

- `fi_domain()` fails with `-38`, or `init_fabric failed: no suitable fabric found`: the step has no CXI service. Add `#SBATCH --network=single_node_vni` (Frontier, any one-node step).
- `VNI_NOT_FOUND` between steps: see [Slingshot VNIs](#slingshot-vnis-and---network).

## Perlmutter filesystems

- Use `$SCRATCH` (Lustre) for `handshake_dir` and `DDSTORE_TEST_DIR`. On CFS, file handshakes across nodes are erratic (slow, or `DDSTORE_HANDSHAKE_TIMEOUT_S` reached).
- torch's `file://` rendezvous fails on CFS from compute nodes (`DistStoreError: Unknown error 524`: no `flock()`). Use `env://` or a `$SCRATCH` path.
