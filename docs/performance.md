# Performance

- Use **batched reads** (the default with `DistDataset`, or `get_batch()` directly). They cut per-sample cost by 10–27× for small rows and make the GPU path insensitive to worker threads; in the VAE every configuration got 1.2–3.9× faster per epoch.
- **Reuse destination buffers** and [`register_recv()`](api-pyddstore.md#register_recvname-arr--unregister_recvname-arr) them. Reading into a fresh buffer every time re-registers memory on every read; `get_profile(name)["mr_miss"]` counts those registrations.
- `method=1` (one-sided `fi_read`) is the fastest backend; `method=0` with batching (collective) comes close for small rows.
- `DDSTORE_PROFILE=1` + `get_profile(name)` shows where `get()`/`get_batch()` time goes: lock wait, memory registration, posting and completing `fi_read`, GPU sync. `vae-ddp.py` prints an all-rank summary when it is set. [examples/scripts/bench_get.py](https://github.com/ORNL/DDStore/blob/main/examples/scripts/bench_get.py) measures per-row latency and throughput vs row size, destination, batch size and threads.

Measurements, profiles and the experiments behind these choices: [docs/results.md](results.md).
