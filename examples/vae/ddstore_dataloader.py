import logging
import os
import socket
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _DatasetKind

logger = logging.getLogger(__name__)


class ThreadDataLoader(DataLoader):
    """DataLoader that parallelizes __getitem__ across a thread pool instead
    of forked worker processes. Threads share the parent's CUDA context and
    Python objects directly, so GPU-resident buffers (DDStore's --gpu-dest/
    --gpu-source path) stay safe across workers -- the default DataLoader's
    forked processes cannot own GPU state, which is why it's capped at
    num_workers=0 for that path.
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self.dataset,
            self._auto_collation,
            self.collate_fn,
            self.drop_last,
        )

        self.fs = queue.Queue()
        # Persistent across epochs -- recreating the pool in every __iter__()
        # would leak OS threads since the old pool is never shut down.
        self._counter = mp.Value("i", 0)
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_workers or 1,
            initializer=self.worker_init,
            initargs=(self._counter,),
        )

        logger.debug("num_workers: %s", self.num_workers)
        logger.debug("len: %s", len(self._index_sampler))

    @staticmethod
    def worker_init(counter):
        core_width = int(os.environ.get("DDSTORE_AFFINITY_WIDTH", "0"))
        core_offset = int(os.environ.get("DDSTORE_AFFINITY_OFFSET", "0"))
        if core_width <= 0 or not hasattr(os, "sched_getaffinity"):
            return 0

        with counter.get_lock():
            wid = counter.value
            counter.value += 1

        affinity = list(os.sched_getaffinity(0))
        affinity_mask = set(
            affinity[
                core_width * wid + core_offset : core_width * (wid + 1) + core_offset
            ]
        )
        if affinity_mask:
            os.sched_setaffinity(0, affinity_mask)
        hostname = socket.gethostname()
        logger.debug(
            "Worker: pid=%s hostname=%s ID=%s affinity=%s",
            os.getpid(),
            hostname,
            wid,
            os.sched_getaffinity(0),
        )
        return 0

    @staticmethod
    def fetch(dataset, ibatch, index, collate_fn=None, pin_memory=False):
        # Collate here, in the worker, before pinning: pinning per-sample
        # tensors and collating afterwards would just torch.stack them into
        # a new, unpinned tensor.
        batch = [dataset[i] for i in index]
        if collate_fn is not None:
            batch = collate_fn(batch)
        if pin_memory:
            batch = torch.utils.data._utils.pin_memory.pin_memory(batch)
        return (ibatch, batch)

    def __iter__(self):
        if self.fs.qsize() > 0:
            for future in iter(self.fs.get, None):
                future.cancel()

        self._num_yielded = 0
        self._sampler_iter = iter(self._index_sampler)
        self.fs_iter = iter(self.fs.get, None)
        self._next_batch_i = 0
        self._inflight = 0
        self._sampler_exhausted = False
        # Bound how many batches can be in flight (submitted but not yet
        # consumed via __next__) at once, instead of submitting the whole
        # epoch up front -- keeps memory use (GPU tensors included) bounded
        # regardless of dataset size. Mirrors torch's own prefetch_factor
        # (default 2 per worker).
        self._max_inflight = max(1, (self.num_workers or 1) * (self.prefetch_factor or 2))
        self._refill()
        return self

    def _refill(self):
        while self._inflight < self._max_inflight:
            try:
                index = next(self._sampler_iter)
            except StopIteration:
                if not self._sampler_exhausted:
                    self._sampler_exhausted = True
                    self.fs.put(None)
                return
            future = self.executor.submit(
                self.fetch,
                self.dataset,
                self._next_batch_i,
                index,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
            )
            self.fs.put(future)
            self._next_batch_i += 1
            self._inflight += 1

    def __next__(self):
        # Refill *before* popping this call's batch, not after: refilling
        # here only uses capacity freed by the *previous* call's batch,
        # which -- by ordinary for-loop semantics -- the caller's loop body
        # has already fully consumed by the time it asks for the next item
        # (i.e. calls __next__ again). Bounds how far the executor can race
        # ahead of consumption (memory, not correctness -- distdataset.py's
        # get() allocates a fresh destination per call, nothing shared to
        # race on).
        self._refill()
        future = next(self.fs_iter)
        ibatch, data = future.result()
        self._inflight -= 1
        self._num_yielded += 1
        return data

    def clean(self):
        if self.fs.qsize() > 0:
            for future in iter(self.fs.get, None):
                future.cancel()

    def __del__(self):
        self.clean()
        self.executor.shutdown(wait=False)
