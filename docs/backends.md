# Backends

## MPI RMA (`method=0`, default)

Uses `MPI_Win_create` and `MPI_Get` for one-sided remote reads. Works on any MPI-capable cluster without additional hardware. `epoch_begin`/`epoch_end` are required to delimit access epochs.

## libfabric RDMA (`method=1`)

Uses `fi_read` for true RDMA transfers over high-speed interconnects (Infiniband/verbs, Cray GNI, Intel PSM2, Cray Slingshot). Lower latency than MPI RMA on supported hardware. `epoch_begin`/`epoch_end` are no-ops with this backend.

**`DDSTORE_FABRIC`** selects which libfabric provider to open, for `method=1`/`2`:

- `hsn` (default, unset) — opens the `tcp;ofi_rxm` domain over Cray Slingshot (Frontier).
- `cxi` — opens the native `cxi` domain over Cray Slingshot (Frontier and Perlmutter; Perlmutter is CXI-only). Required for [GPUDirect RDMA](gpudirect.md), and the default in the Frontier job scripts.

The two are independent code paths (not runtime auto-detection), so set this explicitly per system rather than relying on a guess:

```bash
export DDSTORE_FABRIC=hsn   # tcp;ofi_rxm (default)
export DDSTORE_FABRIC=cxi   # native CXI: Perlmutter, or Frontier with GPUDirect
```

`PyDDStore` picks the network interface (`FABRIC_IFACE`) automatically for `method=1`/`2`, based on each rank's real CPU affinity (`os.sched_getaffinity`) — no changes needed in your code:

- **`DDSTORE_NIC_MAP`** — a precomputed CPU→NIC map, used directly if set (no NIC discovery at construction time). Generate it once from a context with reliable NIC visibility, e.g. an `sbatch` batch step's own shell (not a nested `srun` task — NIC/PCI discovery has been observed to fail there), and export it before launching ranks so every one inherits it:
  ```bash
  export DDSTORE_NIC_MAP=$(python3 -m cpu_nic_map --env)
  srun ... python train.py
  ```
- If `DDSTORE_NIC_MAP` isn't set, each rank falls back to a live `hwloc-calc`/`lstopo` query against its own CPU affinity (`cpu_nic_map.allocated_nics()`, also runnable standalone as `python3 cpu_nic_map.py --allocated`) to find the nearest NIC.
- Set `FABRIC_IFACE` explicitly to override both and force a specific interface, e.g. when the automatic selection picks the wrong one:
  ```bash
  export FABRIC_IFACE=hsn0   # e.g. Cray Slingshot
  ```
- Or skip the environment entirely and pass a map straight to the constructor: `PyDDStore(comm, method=1, nic_map="hsn0=0-15,64-79;hsn1=...")`.

## File-based handshake (`method=2`)

Splits the dataset-holding job from the training job entirely: a **core** group loads and publishes data, and a separate **extra** group reads it over RDMA (`fi_read`, same transport as `method=1`) — the two are independent MPI jobs (e.g. two separate `srun`/`mpirun` launches, possibly on different node allocations) that never share a communicator. They rendezvous only through record files written to a shared-filesystem directory (must be visible to all nodes, e.g. Lustre):

- **Core member** — has an MPI communicator, publishes with `add()`/`init()`. Core ranks exchange records via `MPI_Allgather`, and rank 0 writes the combined set to a single `{name}.bin` file (fabric address, MR key, base pointer, row count, dtype per rank) into `handshake_dir`.
- **Extra member** — no MPI communicator; constructed with `comm_or_none=None` and an explicit `n_core`. Calls `join(name)` to poll for and read all `n_core` core-rank records, then `get()` works exactly as on the core side, reading directly from core-rank memory over RDMA.

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

Environment variables: `DDSTORE_HANDSHAKE_DIR`, `DDSTORE_HANDSHAKE_TIMEOUT_S`, `DDSTORE_FABRIC` and `DDSTORE_NIC_MAP` — see [Environment variables](environment.md).

See [test/test_method2_core.py](https://github.com/ORNL/DDStore/blob/main/test/test_method2_core.py) / [test/test_method2_extra.py](https://github.com/ORNL/DDStore/blob/main/test/test_method2_extra.py) for a minimal runnable pair, and [examples/vae/vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py) / [examples/vae/vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) for a full DDP training example using this split.

`ddstore_width` grouping (below) is not currently supported with `method=2` — every core rank in `comm` is treated as one group.


## Partitioned / Sub-communicator Usage

`PyDDStore` always spans the whole communicator you pass it. To run several independent stores side by side (e.g. one per node), split `comm` first; each group then holds a full replica of the dataset, partitioned across its own members:

```python
width = 4                                       # ranks per group, e.g. GPUs per node
sub_comm = comm.Split(rank // width, rank)
store = dds.PyDDStore(sub_comm)                  # one independent store per group
```

This keeps sample fetches inside a node at the cost of replicating the data per group. `DistDataset` exposes it as `ddstore_width` (`None` = one store across all ranks). Not supported with `method=2`.
