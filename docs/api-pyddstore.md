# `PyDDStore` reference

## `PyDDStore(comm_or_none=None, method=0, handshake_dir="", n_core=0, nic_map=None, rank=None, size=0, exchange_ns="", allgather=None)`

| Parameter | Type | Description |
|---|---|---|
| `comm_or_none` | `mpi4py.MPI.Comm` or `None` | MPI communicator covering all ranks. `None` for a `method=2` extra member, or for a store [without MPI](backends.md#without-mpi-methods-1-and-2) (with `rank=`) |
| `method` | `int` | `0` = MPI RMA (default), `1` = libfabric RDMA, `2` = file-based handshake (see [below](backends.md#file-based-handshake-method2)) |
| `handshake_dir` | `str` | Required for `method=2`, and without MPI unless `allgather` is given: shared-filesystem directory used to exchange RDMA addresses |
| `n_core` | `int` | Required for a `method=2` extra member: number of core ranks that published data. Without MPI, also accepted for `size` with `method=2` |
| `rank`, `size` | `int` | Without MPI (`comm_or_none=None`, `method=1`/`2`): this process is rank `rank` of `size` (e.g. `SLURM_PROCID`, `SLURM_NTASKS`) |
| `exchange_ns` | `str` | Without MPI: names this store's exchange files; default `s0`, `s1`, … in creation order, which is right as long as every rank creates its stores in the same order. Avoid `.` in it |
| `allgather` | callable or `None` | Without MPI: exchange records through `allgather(data: bytes)`, which returns every rank's bytes by rank (a list, or their concatenation), instead of files. All ranks call it together in `add()`/`init()` (e.g. `torch.distributed`); its exception is re-raised there and should occur on every rank |
| `nic_map` | `str` or `None` | Optional, `method=1`/`2` only: a precomputed CPU→NIC map string (see [`DDSTORE_NIC_MAP`](backends.md#libfabric-rdma-method1) below) to use instead of the environment variable. Ignored if `FABRIC_IFACE` is already set |

Call shapes:

```python
PyDDStore(comm)                                     # method 0, MPI RMA
PyDDStore(comm, method=1)                            # method 1, libfabric RDMA
PyDDStore(comm, method=2, handshake_dir="/path")      # method 2, core member (n_core == comm size)
PyDDStore(None, method=2, handshake_dir="/path", n_core=N)  # method 2, extra member (no comm)
PyDDStore(None, method=1, handshake_dir="/path", rank=r, size=N)  # method 1 without MPI
PyDDStore(None, method=2, handshake_dir="/path", rank=r, size=N)  # method 2 core member without MPI
PyDDStore(None, method=1, rank=r, size=N, allgather=fn)  # method 1 without MPI, no files
```

In a build without MPI (`DDSTORE_MPI=0`, `pyddstore.HAVE_MPI` is `False`) only the last three work; a communicator raises `RuntimeError`.

Independent stores per group of ranks: `comm.Split()` and pass the sub-communicator (`DistStore(ddstore_width=)` does this).

---

## `init(name, nrows, disp, itemsize=1)`

Pre-allocate a named variable without providing data yet. Use `update()` to fill it in afterwards. **Collective**. Raises `ValueError` if `name` is in use (as do `add()` and `join()`).

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `nrows` | `int` | Number of rows in this rank's shard |
| `disp` | `int` | Number of elements per row |
| `itemsize` | `int` | Bytes per element (default `1`) |

---

## `add(name, arr)`

Register a NumPy array as a named variable. Each rank contributes its local shard; the global index space is the concatenation of all shards in rank order. **Collective** — all ranks in `comm` must call with the same `name`.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` or `torch.Tensor` | C-contiguous 2-D (or 1-D) array/tensor. Supported dtypes: `int32`, `int64`, `uint8`, `float32`, `float64`, `bool_`/`bool`. A CUDA/HIP tensor registers GPU memory directly — see [GPUDirect RDMA](gpudirect.md) below |

---

## `update(name, arr, offset)`

Overwrite a region of the local shard for a variable registered with `init()`. Local operation — does not require epoch or barrier.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` | Data to write |
| `offset` | `int` | Row offset within the local shard |

---

## `get(name, arr, start=0)`

Read `arr.shape[0]` consecutive rows starting at global index `start` into `arr`. The range must fall within a single rank's shard. With `method=0`, call it between `epoch_begin()` and `epoch_end()` so that other ranks' `update()` writes are visible.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` or `torch.Tensor` | Pre-allocated, C-contiguous output buffer. A CUDA/HIP tensor writes the RDMA transfer directly into GPU memory — see [GPUDirect RDMA](gpudirect.md) below |
| `start` | `int` | Global row index |

---

## `get_batch(name, arr, indices)`

Read rows `indices` (global ids; any order, repeats allowed) into `arr`; row `i` of `arr` gets row `indices[i]`, so `arr.shape[0] == len(indices)` (else `ValueError`). Buffers as for `get()`. Indices are checked first; an out-of-range one raises `IndexError`.

`method=1`/`2`: all rows' `fi_read`s are in flight together, with one lock acquisition, at most one memory registration (none into a [registered](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr) buffer) and one device sync (GPU destination). Rows over `DDSTORE_MAX_READ_BYTES` (1 GiB) are split.

`method=0`: **collective**, after [MDLoader](https://ieeexplore.ieee.org/abstract/document/10820758) ([Citation](citation.md)): indices are all-gathered (`MPI_Allgatherv`) and rows delivered by `MPI_Alltoallv`, in rounds of at most `DDSTORE_ALLTOALL_MAX_BYTES` (2 MiB) per rank. Every rank calls it the same number of times, in the same order, from one thread; index counts may differ (including 0). A bad index raises on every rank. `DistributedSampler` meets this.

```python
idx = np.array([2048, 7, 4096, 7])
out = np.zeros((len(idx), 64), dtype=np.float32)
store.get_batch("features", out, idx)
```

`DistDataset`/`DistDatasetReader` use it through `__getitems__` by default; `DDSTORE_BATCH_GET=0` reads per sample.

---

## `register_recv(name, arr)` / `unregister_recv(name, arr)`

Register a reused destination (C-contiguous array or CUDA/HIP tensor) once, so reads into it or any slice skip memory registration, which can cost more than the transfer for large rows. Several buffers per variable; none evicted. The store keeps a reference until `unregister_recv()` or `free()`. No-op for `method=0`.

```python
pool = np.empty((batch_size, ncols), dtype=np.float32)
store.register_recv("features", pool)
for idx in batches:
    store.get_batch("features", pool[: len(idx)], idx)   # no registration
```

---

## `get_profile(name)`

With `DDSTORE_PROFILE=1`: `get()`/`get_batch()` timing for `name` (`method=1`/`2`), seconds unless noted. Per variable: `calls`, `rows`, `lock_wait`, `mr` (registration), `mr_miss` (count), `read` (posting `fi_read`), `cq` (completion wait). Per store: `py_gets`, `py_get` (whole calls), `py_sync` (GPU sync). Zero for `method=0` or without profiling. See [Performance](performance.md).

---

## `join(name)`

`method=2` extra member: wait for the core group's `{name}.bin` record file (up to `DDSTORE_HANDSHAKE_TIMEOUT_S`), then read the variable with `get()`/`get_batch()`.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier, matching the `name` used in the core group's `add()` |

---

## `fork_reader()`

A read-only `PyDDStore` for the same variables, with its own libfabric endpoints. Methods 1 and 2.

- **Forked process** (`DataLoader` worker): call it in the child; host buffers only. On CXI needs `DDSTORE_FORK_WORKERS=1` before the parent's store is created (raises under `kdreg2`).
- **Same process**: one reader per thread, so threads read in parallel.

`DistDataset` and `DistDatasetReader` use it automatically. See [Forked DataLoader workers](pytorch.md#forked-dataloader-workers).

---

## `has(name)`

`True` if variable `name` is added or joined and not freed.

---

## `info(name)`

Returns `(total_rows, disp, itemsize)` for a variable that has been `add()`-ed or `join()`-ed. Useful on the extra side to size output buffers without hardcoding shapes.

---

## `epoch_begin()` / `epoch_end()`

**Collective** synchronization points for `method=0`: `epoch_begin()` makes each rank's `update()` writes visible and waits for every rank; `epoch_end()` waits until every rank is done reading. Each `get()` locks its target itself (`MPI_Win_lock`), so these hold no RMA epoch. No-op for `method=1`/`2`.

---

## `free()`

Release MPI windows or libfabric endpoints and registrations (including `register_recv()` buffers), and host buffers from `add()`/`init()`; GPU tensors passed to `add()` stay the caller's. Idempotent; skips MPI resources after `MPI_Finalize`. Freed names can be added again.

---

## `pyddstore.HAVE_MPI`

`False` for a `DDSTORE_MPI=0` build (methods 1 and 2 without a communicator).

---

## `pyddstore.FileComm(dir, rank, size, ns)`

File-based collectives without MPI: `allgather(obj)` (list by rank) and `barrier()`. `dir` on a shared filesystem; every rank creates its `FileComm`s and makes calls in the same order, with an `ns` unique in `dir` (also among stores' `exchange_ns`). Attributes: `rank`, `size`, `dir`, `ns`. Rank 0 leaves `barrier()` once it has written the combined file, so don't remove `dir` right after one.
