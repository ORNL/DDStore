# DDStore

<img src="../images/DDStore-logo.png" alt="DDStore logo" />

Efficient distributed data loading for distributed data-parallel (DDP) training.

Each MPI rank holds a shard of the full dataset in memory. DDStore exposes a global index space so any rank can read any sample via one-sided remote memory access — either MPI RMA (default) or libfabric RDMA — without coordinator synchronization.

- **Batched reads**: [`get_batch()`](api-pyddstore.md#get_batchname-arr-indices) fetches a whole training batch in one call (one-sided RDMA reads in flight together, or an MPI collective for `method=0`).
- **GPUDirect RDMA**: data can live in, and be read straight into, GPU memory ([details](gpudirect.md)).
- **PyTorch integration**: [`pyddstore.torch`](pytorch.md) turns any map-style dataset into a distributed one (`DistDataset`) and provides a thread-based `ThreadDataLoader` that is safe with MPI and GPU buffers.
- **Thread-safe** reads, a [profiler](performance.md) for where read time goes, and a split mode (`method=2`) where a separate job reads data published by another.

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="DDStore architecture" width="300" />

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 2

backends
gpudirect
pytorch
hpc
performance
concurrency
```

```{toctree}
:caption: Reference
:maxdepth: 2

api-pyddstore
api-torch
environment
```

```{toctree}
:caption: More
:maxdepth: 1

testing
results
citation
```
