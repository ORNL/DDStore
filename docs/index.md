# DDStore

<img src="../images/DDStore-logo.png" alt="DDStore logo" />

Efficient distributed data loading for distributed data-parallel (DDP) training.

Each rank holds a shard of the dataset in memory; any rank reads any sample by global index through one-sided remote memory access (MPI RMA or libfabric RDMA), without coordination.

- **Batched reads**: [`get_batch()`](api-pyddstore.md#get_batchname-arr-indices) reads a whole batch in one call.
- **GPUDirect RDMA**: data can live in, and be read straight into, GPU memory ([details](gpudirect.md)).
- **PyTorch integration**: [`pyddstore.torch`](pytorch.md): `DistDataset` (any map-style dataset), `DistStore` (several datasets on one store), `WindowedDataset` (multi-row samples), `ThreadDataLoader` (threads; safe with MPI and GPU buffers). PyTorch's `DataLoader` with [forked workers](pytorch.md#forked-dataloader-workers) also works (methods 1 and 2).
- **Thread-safe** reads, a read-time [profiler](performance.md), and a split mode (`method=2`) where one job reads data published by another.
- **MPI optional**: methods 1 and 2 also run [without MPI](backends.md#without-mpi-methods-1-and-2) (`DDSTORE_MPI=0`), launched by `srun`.

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
