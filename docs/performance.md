# Performance

- **Batched reads** (`get_batch()`, the `DistDataset` default): 10–27× cheaper per row for small rows; 1.2–3.9× faster VAE epochs.
- **Loader workers**: one or two `ThreadDataLoader` workers usually hide batched reads; forked `DataLoader` workers need `persistent_workers=True`.
- **Reuse destination buffers** with [`register_recv()`](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr); fresh buffers are registered on every read (`get_profile(name)["mr_miss"]`).
- `method=1` is fastest; batched `method=0` comes close for small rows.
- **Profiling**: `DDSTORE_PROFILE=1` + `get_profile(name)` splits read time into lock wait, registration, `fi_read` post and completion, and GPU sync (`vae-ddp.py` prints a summary). [bench_get.py](https://github.com/ORNL/DDStore/blob/main/examples/scripts/bench_get.py) measures per-row cost vs row size, destination, batch size and threads.

Measurements: [results](results.md).
