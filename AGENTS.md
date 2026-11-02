# Notes for AI coding agents

For developing DDStore itself. To add DDStore to an application, use [skills/ddstore](skills/ddstore/SKILL.md).

## Source map

| Path | What |
|---|---|
| `include/ddstore.hpp`, `src/ddstore.cxx` | C++ store: variables, methods 0 (MPI RMA), 1 (libfabric), 2 (libfabric + file handshake) |
| `include/exchange.hpp`, `src/exchange.cxx` | Record exchange: `MpiExchange`, `FileExchange`, `CallbackExchange` |
| `include/common.h`, `src/common.cxx` | libfabric setup, NIC selection |
| `src/pyddstore/_core.pyx` | Cython `PyDDStore`, `FileComm` |
| `src/pyddstore/torch.py` | `DistStore`, `DistDataset`, `DistDatasetReader`, `ThreadDataLoader`, `WindowedDataset` |
| `src/cpu_nic_map.py` | CPU→NIC map from sysfs (`python -m cpu_nic_map --env`) |
| `test/` | pytest; see `docs/testing.md` for ranks and requirements |
| `examples/minimal/`, `examples/vae/` | Runnable training examples and Slurm scripts |
| `docs/` | Sphinx (MyST) docs |
| `skills/ddstore/` | Skill for agents integrating DDStore; `test/test_skill.py` checks it against the code |

## Build

```bash
CC=mpicc CXX=mpicxx pip install --no-build-isolation --no-deps -e .          # MPI build
DDSTORE_MPI=0 CC=gcc CXX=g++ pip install --no-build-isolation --no-deps -e . # no MPI: methods 1, 2
# Frontier: CC=cc CXX=CC, and SETUPTOOLS_USE_DISTUTILS=stdlib if distutils.msvccompiler is missing
```

Switching builds rebuilds only the extension (`setup.py` tracks the mode).

## Invariants

- `add`, `init`, `free`, `DistStore(...)`, `DistDataset(...)` are collective: every rank calls them in the same order. Errors found on some ranks must be raised on all (gather first), or the others hang.
- Method 0 reads (`get_batch`, `read_rows`, `__getitems__`) are collective.
- Forked processes only (`DataLoader` workers); `spawn`/`forkserver` are not supported with MPI on Frontier.
- Without MPI, nothing may import `mpi4py`.
- 2.x is backward compatible: don't remove or reorder public arguments.
- Docs are terse and neutral. A user-visible change updates `docs/`, `README.md` if affected, and `skills/ddstore/` if the API changes.

## Checks

```bash
black --check $(git ls-files '*.py')                       # black 26.5.1, as CI
sphinx-build -W --keep-going -b html docs docs/_build/html  # docs CI fails on warnings
python -m pytest -q test/test_skill.py test/test_cpu_nic_map.py   # no MPI or cluster
mpirun -n 1 python -m pytest -q test/test_single.py
mpirun -n 2 python -m pytest -q test/test_multirank.py test/test_get_batch.py test/test_torch.py
```

CI (GitHub Actions, CPU only) runs method 0 with MPICH and Open MPI and the no-MPI build. Methods 1 and 2 over `cxi`, GPU tests and `test_torch_nompi.py` need a Slurm step on a Slingshot system: `srun`, not `torchrun`, with `--network=single_node_vni` (Frontier) and `DDSTORE_FABRIC=cxi`. Test matrix: `docs/testing.md`.
