# Concurrency

- `get()` and `get_batch()` are thread-safe. For `method=1`/`2` a per-variable lock in `DDStore::get()` serializes calls on one variable (it guards shared receive state; without it concurrent calls crashed). Both release the GIL during the transfer.
- `method=0` `get_batch()` is **collective**: every rank calls it the same number of times, in the same order, from one thread. `vae-ddp.py` therefore allows no worker threads with `method=0`.
- Only the main thread calls MPI (setup, `epoch_begin`/`epoch_end`, `method=0` reads); mpi4py's default `MPI_THREAD_MULTIPLE` is fine, `FUNNELED` is the minimum. If you call MPI from your own worker threads, keep `MULTIPLE`.
