# Concurrency

- `get()` and `get_batch()` are thread-safe and release the GIL. For `method=1`/`2`, calls on one variable of one store take turns (per-variable lock); for parallel reads give each thread a [`fork_reader()`](api-pyddstore.md#fork_reader), as `ThreadDataLoader` does.
- `method=0` `get_batch()` is **collective**: same number of calls, same order, one thread per rank. `vae-ddp.py` allows no workers with `method=0`.
- MPI thread level: `FUNNELED` is enough when only the main thread calls DDStore with `method=0`. `ThreadDataLoader` reads on a worker thread (also with `num_workers=0`), so with `method=0` use one worker and `SERIALIZED` or higher (mpi4py defaults to `MULTIPLE`), and make no MPI calls on the same communicator from other threads while it reads. Methods 1/2 make no MPI calls when reading.
