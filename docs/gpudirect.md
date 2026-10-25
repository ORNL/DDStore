# GPUDirect RDMA

`add()`, `get()` and `get_batch()` accept a CUDA/HIP tensor, so RDMA reads from and into GPU memory without host copies. Requires `method=1`/`2`, **`DDSTORE_FABRIC=cxi`**, and a CUDA or ROCm PyTorch; otherwise a GPU tensor raises.

```python
import torch
data = torch.rand(1024, 64, dtype=torch.float32, device="cuda")
store.add("features", data)                      # GPU source, no host copy

out = torch.empty((1, 64), dtype=torch.float32, device="cuda")
store.get("features", out, start=2048)            # GPU destination, no host copy
```

- **`add()` registers the GPU tensor itself (no copy).** Keep it unmodified until `free()`; the store holds a reference, and re-adding the name with a GPU tensor is rejected. NumPy arrays are copied.
- **One device sync per GPU read** (`torch.cuda.synchronize()` per `get()`/`get_batch()`): the NIC writes outside PyTorch's stream order. Use `get_batch()` so it's one sync per batch.
- `init()`/`update()` stay host-only.
- GPU vs host destinations: on Frontier GPU wins from ~12.5 KB rows, on Perlmutter host wins at every size ([results](results.md#bench_getpy-µs-per-row)).

- Forked `DataLoader` workers read into host memory only ([details](pytorch.md#forked-dataloader-workers)).

Examples: [test/test_gpu_rdma.py](https://github.com/ORNL/DDStore/blob/main/test/test_gpu_rdma.py); `--gpu-dest`/`--gpu-source` on [vae-ddp.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae-ddp.py), [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) and [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py).
