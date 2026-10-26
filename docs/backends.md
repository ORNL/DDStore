# Backends

## MPI RMA (`method=0`, default)

`MPI_Win_create` + `MPI_Get` (passive target: each read locks its target). Runs on any MPI cluster (MPICH, Open MPI). Reads go between `epoch_begin()` and `epoch_end()`, which synchronize ranks so `update()` writes are visible.

## libfabric RDMA (`method=1`)

`fi_read` over high-speed interconnects (Slingshot, InfiniBand/verbs, …); lower latency than MPI RMA. `epoch_begin`/`epoch_end` are no-ops.

**`DDSTORE_FABRIC`** (methods 1 and 2) picks the provider; set it per system:

- `hsn` (default): `tcp;ofi_rxm` over Slingshot (Frontier).
- `cxi`: native Slingshot (Frontier, Perlmutter). Required for [GPUDirect RDMA](gpudirect.md); the Frontier job scripts use it.

The network interface (`FABRIC_IFACE`) is chosen per rank from its CPU affinity:

- **`DDSTORE_NIC_MAP`** (recommended): a precomputed CPU→NIC map, used if set. Generate it once in the batch step's shell and export it:
  ```bash
  export DDSTORE_NIC_MAP=$(python3 -m cpu_nic_map --env)
  srun ... python train.py
  ```
- Otherwise each rank queries `hwloc` for its nearest NIC (`python3 cpu_nic_map.py --allocated` to check). On Frontier compute nodes this took 8–18 s per process, versus under 0.01 s with the map, and NIC discovery can fail inside nested `srun` tasks.
- `FABRIC_IFACE=hsn0` forces an interface; `PyDDStore(..., nic_map="hsn0=0-15,64-79;...")` passes a map directly.

## File-based handshake (`method=2`)

Separates the job holding the data from the training job. A **core** group publishes data; a separate **extra** group (its own launch, possibly other nodes, no shared communicator) reads it over RDMA as in `method=1`. They meet only through record files in a shared directory (e.g. Lustre):

- **Core member**: has a communicator (or `rank=`/`size=` [without MPI](#without-mpi-methods-1-and-2)); publishes with `add()`/`init()`. Rank 0 writes the combined records (address, key, base pointer, rows, dtype per rank) to `{name}.bin` in `handshake_dir`.
- **Extra member**: `comm_or_none=None` and `n_core`; `join(name)` reads the records, then `get()`/`get_batch()` read from core memory.

```python
# core side — one MPI job
store = dds.PyDDStore(comm, method=2, handshake_dir="/lustre/.../ddstore_hs")
store.add("x", data)
...                       # wait for the extra side to finish (e.g. a sentinel file)
store.free()

# extra side — a separate MPI job, no comm needed
store = dds.PyDDStore(None, method=2, handshake_dir="/lustre/.../ddstore_hs", n_core=4)
store.join("x")
out = np.zeros((1, ncols), dtype=np.float32)
store.get("x", out, start=global_idx)
store.free()
```

Settings: `DDSTORE_HANDSHAKE_DIR`, `DDSTORE_HANDSHAKE_TIMEOUT_S`, `DDSTORE_FABRIC`, `DDSTORE_NIC_MAP` ([Environment variables](environment.md)). Examples: [test_method2_core.py](https://github.com/ORNL/DDStore/blob/main/test/test_method2_core.py) / [test_method2_extra.py](https://github.com/ORNL/DDStore/blob/main/test/test_method2_extra.py); [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py) / [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) for DDP training. `ddstore_width` is not supported: all core ranks form one group.

## Without MPI (methods 1 and 2)

Methods 1 and 2 use MPI only to gather each rank's record in `add()`/`init()`. Without a communicator, that gather goes through files in a shared directory:

```python
import os
import pyddstore as dds

rank, size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
store = dds.PyDDStore(None, method=1, handshake_dir="/lustre/.../hs", rank=rank, size=size)
store.add("x", data)                 # collective, through files in handshake_dir
store.get_batch("x", out, indices)   # RDMA reads, as with MPI
store.free()
```

```bash
srun -N2 -n8 python train.py         # no mpirun, no MPI library needed
```

- Works in either build; the only option in a [`DDSTORE_MPI=0` build](installation.md#building-without-mpi).
- `method=2` gives a core member (`size=` is `n_core`) that publishes `{name}.bin` as usual.
- Put `handshake_dir` on Lustre (Perlmutter: `$SCRATCH`, not CFS, where polling across nodes is erratic).
- Every rank creates its stores and calls `add()`/`init()` in the same order. Ranks poll (1–200 ms backoff) for up to `DDSTORE_HANDSHAKE_TIMEOUT_S`.
- File names include a session (`DDSTORE_SESSION`, else `{SLURM_JOB_ID}_{SLURM_STEP_ID}`) and a store name (`exchange_ns`, default `s0`, `s1`, …), and each exchange has its own generation and nonces, so runs and stores can share a directory.
- `epoch_begin`/`epoch_end` are no-ops. For barriers (e.g. before `free()`), use `pyddstore.FileComm(dir, rank, size, ns)`.

`DistDataset` does this when it has no MPI communicator ([PyTorch integration](pytorch.md#without-mpi)).

## Partitioned / Sub-communicator Usage

A store spans its whole communicator. For independent stores (e.g. one per node), split `comm`; each group holds a full replica, partitioned across its members:

```python
width = 4                                       # ranks per group, e.g. GPUs per node
sub_comm = comm.Split(rank // width, rank)
store = dds.PyDDStore(sub_comm)                  # one independent store per group
```

Reads stay within the group at the cost of one replica per group. `DistDataset(ddstore_width=)` does this. Not supported with `method=2`.
