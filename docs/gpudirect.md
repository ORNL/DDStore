# GPUDirect RDMA

`add()`, `get()` and `get_batch()` accept a CUDA/HIP `torch.Tensor` in place of a NumPy array, so RDMA reads from or writes directly into GPU memory, with no `.cpu()`/`.to(device)` copy. Requires `method=1` or `2`, **`DDSTORE_FABRIC=cxi`**, and a CUDA- or ROCm-enabled PyTorch. A GPU tensor with `DDSTORE_FABRIC=hsn` (the default) or `method=0` raises a clear error instead of silently copying through the host.

```python
import torch
data = torch.rand(1024, 64, dtype=torch.float32, device="cuda")
store.add("features", data)                      # GPU source, no host copy

out = torch.empty((1, 64), dtype=torch.float32, device="cuda")
store.get("features", out, start=2048)            # GPU destination, no host copy
```

- **`add()` with a GPU tensor registers your tensor's own memory; no copy is made.** Keep it alive and unmodified until `free()`. `PyDDStore` holds a reference as a safety net, and adding the same name again with a GPU tensor is rejected. (With NumPy, `add()` copies and the array can be reused right away.)
- **The device is synchronized before each GPU transfer** (`torch.cuda.synchronize()`, once per `get()` / `get_batch()` call): the NIC writes outside PyTorch's stream ordering, and without the sync training hit GPU memory faults. Prefer `get_batch()` on the GPU path so this costs one sync per batch, not per sample.
- `init()`/`update()` stay host-only.
- Whether GPU destinations are faster than host ones depends on the machine: on Frontier they win from ~12.5 KB rows up, on Perlmutter host destinations win at every size ([results](results.md#bench_getpy-µs-per-row)).

Examples: [test/test_gpu_rdma.py](https://github.com/ORNL/DDStore/blob/main/test/test_gpu_rdma.py), and `--gpu-dest`/`--gpu-source` on [vae-ddp.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae-ddp.py), [vae_extra_train.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_extra_train.py) and [vae_core_server.py](https://github.com/ORNL/DDStore/blob/main/examples/vae/vae_core_server.py).
