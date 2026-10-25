# Concurrency

- `get()` and `get_batch()` are thread-safe and release the GIL. For `method=1`/`2`, calls on one variable of one store take turns (per-variable lock); for parallel reads give each thread a [`fork_reader()`](api-pyddstore.md#fork_reader), as `ThreadDataLoader` does.
- `method=0` `get_batch()` is **collective**: same number of calls, same order, one thread per rank. `vae-ddp.py` allows no workers with `method=0`.
- Only the main thread calls MPI; `FUNNELED` is the minimum thread level (mpi4py defaults to `MULTIPLE`; keep it if your threads call MPI).
