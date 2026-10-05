# Testing

## Unit tests (pytest)

Install test dependencies:

```bash
pip install pytest pytest-mpi
```

**Single-rank** — no cluster required, covers all dtypes, `add`/`get`/`init`/`update`, and error cases:

```bash
mpirun -n 1 python -m pytest test/test_single.py -v
```

**Multi-rank** — verifies remote reads across all rank pairs and sub-communicator grouping:

```bash
mpirun -n 4 python -m pytest test/test_multirank.py -v
```

**Batched reads and the PyTorch layer** — method 0 everywhere; method 1, method 2 and GPU cases run inside a Slurm step with a CXI device (provider from `DDSTORE_FABRIC`, default `cxi`):

```bash
mpirun -n 4 python -m pytest test/test_get_batch.py test/test_torch.py -v
```

**GPUDirect RDMA** — requires a live `cxi` fabric and a CUDA/HIP GPU per rank (skipped automatically otherwise); see [GPUDirect RDMA](gpudirect.md):

```bash
DDSTORE_FABRIC=cxi mpirun -n 2 python -m pytest test/test_gpu_rdma.py -v
```

| Test file | Min ranks | What is tested |
|---|---|---|
| `test/test_single.py` | 1 | All dtypes, `add`/`get`, `init`/`update`/`get`, error handling, double `free()` |
| `test/test_multirank.py` | 2 (4 recommended) | Remote reads, shard boundaries, multiple variables, `ddstore_width` grouping |
| `test/test_gpu_rdma.py` | 2 | GPU-resident `add()`/`get()` in both directions, both libfabric methods, negative/error cases |
| `test/test_get_batch.py` | 2 (4 recommended) | `get_batch()`: shuffled indices across ranks with repeats, single row, dtypes, error recovery, GPU destination, concurrent threads, registered destination buffers, wide rows (with `DDSTORE_MAX_READ_BYTES=4096` they are read in pieces); method 0, plus method 1 over `cxi` inside a Slurm step |
| `test/test_torch.py` | 2 (4 recommended) | `pyddstore.torch`: tuple/dict/single samples of every field kind, numpy records, loaders vs the plain source, `ThreadDataLoader` iterators and prefetch depth, `read_rows`/`alloc` buffers, `reuse_buffers` (also with GPU buffers), `WindowedDataset`, `row_of`, `encode`/`decode`/`fields`, chunked loading, error handling, `ddstore_width`, GPU placement, `DistDatasetReader` |

## Integration scripts

```bash
# Basic functional test (libfabric, method=1)
mpirun -n 4 python examples/scripts/demo.py

# Integration test with PyTorch DDP (libfabric, method=1)
mpirun -n 4 python examples/scripts/test.py
```

Optional arguments for `examples/scripts/demo.py` and `examples/scripts/test.py`:

| Flag | Default | Description |
|---|---|---|
| `--num` | `1048576` | Rows per rank |
| `--dim` | `64` | Elements per row |
| `--nbatch` | `32` | Number of random reads |
| `--gloo` / `--nccl` | `--gloo` | `test.py` only: `torch.distributed` backend |

## Method 2 (file-based handshake)

Two separate launches sharing a handshake directory on a shared filesystem — not a single `mpirun`, since core and extra are independent jobs:

```bash
# Terminal 1 — core (data-holding) side
mpirun -n 4 python test/test_method2_core.py /path/to/shared/ddstore_hs

# Terminal 2 — extra (reader) side, after or while the core side is running
python test/test_method2_extra.py /path/to/shared/ddstore_hs 4
```
