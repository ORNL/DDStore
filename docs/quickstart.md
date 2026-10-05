# Quick start

```python
import numpy as np
from mpi4py import MPI
import pyddstore as dds

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each rank contributes its own shard
store = dds.PyDDStore(comm)                  # MPI RMA backend (default)
# store = dds.PyDDStore(comm, method=1)      # libfabric RDMA backend

data = np.random.rand(1024, 64).astype(np.float32)
store.add("features", data)                  # collective — all ranks must call

# Read any global sample index
out = np.zeros((1, 64), dtype=np.float32)
store.epoch_begin()
store.get("features", out, start=2048)       # global index across all shards
store.epoch_end()

store.free()
```

Run with:
```bash
mpirun -n 4 python my_script.py
```

With PyTorch, [`pyddstore.torch.DistDataset`](pytorch.md) does the sharding, `add()` and batched reads for you.
