# `PyDDStore` reference

## `PyDDStore(comm_or_none=None, method=0, handshake_dir="", n_core=0, nic_map=None)`

| Parameter | Type | Description |
|---|---|---|
| `comm_or_none` | `mpi4py.MPI.Comm` or `None` | MPI communicator covering all ranks. `None` only for a `method=2` extra member |
| `method` | `int` | `0` = MPI RMA (default), `1` = libfabric RDMA, `2` = file-based handshake (see [below](backends.md#file-based-handshake-method2)) |
| `handshake_dir` | `str` | Required for `method=2`: shared-filesystem directory used to exchange RDMA addresses |
| `n_core` | `int` | Required for a `method=2` extra member: number of core ranks that published data |
| `nic_map` | `str` or `None` | Optional, `method=1`/`2` only: a precomputed CPU→NIC map string (see [`DDSTORE_NIC_MAP`](backends.md#libfabric-rdma-method1) below) to use instead of the environment variable. Ignored if `FABRIC_IFACE` is already set |

Four call shapes:

```python
PyDDStore(comm)                                     # method 0, MPI RMA
PyDDStore(comm, method=1)                            # method 1, libfabric RDMA
PyDDStore(comm, method=2, handshake_dir="/path")      # method 2, core member (n_core == comm size)
PyDDStore(None, method=2, handshake_dir="/path", n_core=N)  # method 2, extra member (no comm)
```

Note: grouping ranks into independent stores (the "sub-communicator" pattern below) is done by splitting `comm` yourself before constructing `PyDDStore` — there is no `ddstore_width` constructor parameter. `pyddstore.torch.DistDataset` does this for you (`ddstore_width`) and shows the pattern (`comm.Split()` then `PyDDStore(sub_comm)`).

---

## `init(name, nrows, disp, itemsize=1)`

Pre-allocate a named variable without providing data yet. Use `update()` to fill it in afterwards. **Collective**.

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

Read `arr.shape[0]` consecutive rows starting at global index `start` into `arr`. The range must fall within a single rank's shard. Must be called inside an `epoch_begin` / `epoch_end` pair when using the MPI backend.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier |
| `arr` | `np.ndarray` or `torch.Tensor` | Pre-allocated, C-contiguous output buffer. A CUDA/HIP tensor writes the RDMA transfer directly into GPU memory — see [GPUDirect RDMA](gpudirect.md) below |
| `start` | `int` | Global row index |

---

## `get_batch(name, arr, indices)`

Read rows `indices` (global row ids; any order, any ranks, repeats allowed) into `arr`: row `i` of `arr` receives row `indices[i]`, so `arr.shape[0]` must equal `len(indices)` (else `ValueError`). Same buffer rules as `get()` (NumPy array or CUDA/HIP tensor). Every index is checked before anything is read; an out-of-range one raises `IndexError` and leaves the store usable.

For `method=1`/`2` the whole batch is one call: one lock acquisition, at most one memory registration (none into a [registered](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr) buffer), and (GPU destination) one device sync, with all of the batch's `fi_read`s posted before any is waited for, so the reads overlap on the network. It is one `fi_read` per row, or several for a row longer than `DDSTORE_MAX_READ_BYTES` (default 1 GiB).

For `method=0`, `get_batch()` is **collective**, after the collective module of [MDLoader](https://ieeexplore.ieee.org/abstract/document/10820758) (see [Citation](citation.md)): every rank all-gathers all ranks' indices (`MPI_Allgatherv`), packs the rows it owns for each requester, and one `MPI_Alltoallv` delivers them, on a private duplicate of the store's communicator, in rounds of at most `DDSTORE_ALLTOALL_MAX_BYTES` (default 2 MiB) received per rank so large rows don't turn into one huge exchange. So every rank must call it for the variable the same number of times, in the same order, from one thread at a time; the number of indices may differ per rank (including 0). Indices are checked on the gathered list, so a bad index raises on every rank together. `DistributedSampler` gives every rank the same number of batches, and `vae-ddp.py` allows no worker threads with `method=0`, so the data loaders meet this automatically.

```python
idx = np.array([2048, 7, 4096, 7])
out = np.zeros((len(idx), 64), dtype=np.float32)
store.get_batch("features", out, idx)
```

`DistDataset`/`DistDatasetReader` use it by default through `__getitems__`, which PyTorch's `DataLoader` (and `ThreadDataLoader`) calls with a whole batch's indices, so the VAE examples and job scripts read in batches with no extra flag. Set `DDSTORE_BATCH_GET=0` to fall back to one `get()` per sample.

---

## `register_recv(name, arr)` / `unregister_recv(name, arr)`

Register `arr` (a C-contiguous NumPy array or CUDA/HIP tensor) once as a destination for `get()`/`get_batch()` of `name`. Reads into `arr` or any slice of it then skip memory registration, which otherwise happens whenever the destination isn't the buffer registered by the previous read. For large rows, registration can cost more than the transfer. Use it for buffers you reuse, such as a pool per loader thread: several can be registered per variable and none is evicted. The store holds a reference to `arr` until `unregister_recv()` or `free()`. No-op for `method=0`.

```python
pool = np.empty((batch_size, ncols), dtype=np.float32)
store.register_recv("features", pool)
for idx in batches:
    store.get_batch("features", pool[: len(idx)], idx)   # no registration
```

---

## `get_profile(name)`

With `DDSTORE_PROFILE=1` set before the process starts: timing of `get()`/`get_batch()` for `name` (`method=1`/`2`), in seconds unless noted. C++ counters for this variable: `calls` (get + get_batch), `rows`, `lock_wait`, `mr` (receive-buffer registration, including cache checks), `mr_miss` (registrations, a count), `read` (posting `fi_read`), `cq` (waiting for completions). Python counters for the whole store: `py_gets`, `py_get` (whole calls), `py_sync` (`torch.cuda.synchronize()` on the GPU path). All zero for `method=0` or without profiling. See [Performance](performance.md).

---

## `join(name)`

`method=2` extra member only. Discovers a variable published by the core group by polling the handshake directory until the combined record file (`{name}.bin`) written by core rank 0 reaches its expected size (up to `DDSTORE_HANDSHAKE_TIMEOUT_S` seconds), then registers it for `get()`.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Variable identifier, matching the `name` used in the core group's `add()` |

---

## `info(name)`

Returns `(total_rows, disp, itemsize)` for a variable that has been `add()`-ed or `join()`-ed. Useful on the extra side to size output buffers without hardcoding shapes.

---

## `epoch_begin()` / `epoch_end()`

Open and close an MPI RMA access epoch (calls `MPI_Win_fence`). **Collective**. Required around `get()` calls when using `method=0`. No-op for `method=1`/`2`.

---

## `free()`

Release every variable's MPI window (`method=0`) or libfabric endpoints and memory registrations, including [`register_recv()`](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr) buffers (`method=1`/`2`), then the host buffer DDStore allocated for it in `add()`/`init()` (a GPU tensor passed to `add()` is the caller's and is not freed). Safe to call more than once. After `MPI_Finalize` the MPI window and buffer can no longer be released and are skipped.
