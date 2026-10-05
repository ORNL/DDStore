# HPC systems (Slurm, Slingshot)

Running DDStore under Slurm on Cray Slingshot systems (Frontier, Perlmutter):
the example job scripts, which `--network` options a layout needs, and what
to do when RDMA can't connect.

## Do you need `--network` flags?

Every `srun` step gets its own Slingshot VNI (network isolation ID), and two
endpoints can only reach each other on the same VNI. Whether you need
`#SBATCH --network=single_node_vni,job_vni` depends on how steps talk:

| Layout | Perlmutter, no flags | Perlmutter, `single_node_vni,job_vni` | Frontier |
|---|---|---|---|
| One `srun` step (any number of nodes): normal training, `method=0`/`1` | works | works | works; `single_node_vni` needed if the step has one node |
| Core and extra as separate one-node steps (`--layout=split-node`) | works: one-node steps get no VNI and share the default one | needs `job_vni` + the wrapper below | needs both flags + the wrapper |
| Core and extra on the same nodes, or steps spanning several nodes | extra can't reach core | needs `job_vni` + the wrapper + `srun --overlap` | second step fails to launch |

The Perlmutter columns were measured on 2 nodes (job scripts and logs in the
[results](results.md#perlmutter-validation-2-nodes--4-a100-cuda-13-cxi)).
## Slurm job scripts

[job-vae-single.sh](https://github.com/ORNL/DDStore/blob/main/examples/vae/script/job-vae-single.sh) runs `vae-ddp.py` as one `srun` step; [job-vae-core-extra.sh](https://github.com/ORNL/DDStore/blob/main/examples/vae/script/job-vae-core-extra.sh) runs the core/extra split as two steps. Run either with `--help` for all options. Their `#SBATCH` lines target Frontier (`-A FUS184`, 8 ranks × 7 cores per node); see below for Perlmutter.

```bash
sbatch examples/vae/script/job-vae-single.sh --method=1 --num-workers=1
sbatch examples/vae/script/job-vae-single.sh --method=1 --gpudirect --image-scale=2
sbatch examples/vae/script/job-vae-core-extra.sh                       # split-node: 1 core node, the rest extra
sbatch examples/vae/script/job-vae-core-extra.sh --gpudirect --core-nnodes=2
```

`job-vae-core-extra.sh` sets up Slingshot networking for its two steps (see [Multiple `srun` steps](hpc.md#multiple-srun-steps-in-one-job-method2-cxi)). `--layout=colocate` (both steps on the same nodes) works on Perlmutter only.

Both scripts also run on Perlmutter: they detect the machine (`NERSC_HOST`) and use 4 ranks per node, with the training ranks seeing all 4 GPUs of their node (NCCL needs that there). Override the Frontier `#SBATCH` lines when submitting:

```bash
sbatch -A <account> -C gpu --gpus-per-node=4 examples/vae/script/job-vae-single.sh --method=1
```


## Multiple `srun` steps in one job (`method=2`, `cxi`)

On Slingshot every `srun` step gets its own VNI (network isolation ID), and two endpoints can only talk on the same VNI. For core and extra running as separate steps, [job-vae-core-extra.sh](https://github.com/ORNL/DDStore/blob/main/examples/vae/script/job-vae-core-extra.sh) does both of these:

1. `#SBATCH --network=single_node_vni,job_vni`: `job_vni` adds a job-wide VNI to every step (`SLINGSHOT_VNIS=<step VNI>,<job VNI>`); on Frontier, `single_node_vni` is also what gives single-node steps a CXI service at all (without it `fi_domain()` fails with `-38`).
2. In each task, before Python starts: `export SLINGSHOT_VNIS=${SLINGSHOT_VNIS##*,}`. libfabric's cxi provider uses only the first VNI listed (the step's own), so without this reads fail with `VNI_NOT_FOUND`.

Colocating both steps on the same nodes:

| | `job_vni` + wrapper | `job_vni` + wrapper + `srun --overlap` | no `--network` |
|---|---|---|---|
| Frontier | second step fails to launch: `Error configuring interconnect` | same failure | each step has only its own VNI: extra cannot reach core |
| Perlmutter | second step does not start | **works** | extra cannot reach core |

So the script defaults to `--layout=split-node`; `--layout=colocate` (with `--overlap`) is for Perlmutter. Perlmutter's single-node steps also work without the `--network` flags.


## Troubleshooting: RDMA fails to connect (`cxi`)

If `fi_domain()` fails with `-38 (Function not implemented)`, the step has no CXI service: add `#SBATCH --network=single_node_vni` (needed on Frontier for any single-node step, i.e. a `-N 1` job or a one-node step inside a larger job). If ranks in different `srun` steps can't reach each other (`VNI_NOT_FOUND`), see [Multiple `srun` steps](hpc.md#multiple-srun-steps-in-one-job-method2-cxi) above.
