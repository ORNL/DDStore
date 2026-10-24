"""DDStore: distributed in-memory data store for data-parallel training.

The store itself is ``PyDDStore`` (compiled extension ``pyddstore._core``).
PyTorch helpers (``DistDataset``, ``DistDatasetReader``, ``ThreadDataLoader``)
live in ``pyddstore.torch`` and need PyTorch; importing ``pyddstore`` alone
does not.

``pyddstore.HAVE_MPI`` is False for a build made with ``DDSTORE_MPI=0``:
only methods 1 and 2 without a communicator (``rank=``/``size=``) work there.

The extension is loaded on first use of ``PyDDStore`` or ``HAVE_MPI``, not
on ``import pyddstore``. It does not import mpi4py itself: MPI starts when
the caller imports mpi4py to make the communicator it passes. Import torch
before that (``from pyddstore.torch import ...`` does), the order torch +
RCCL/NCCL need to shut down cleanly.

Methods 1 and 2 can be read from forked PyTorch DataLoader workers
(``num_workers > 0``): each worker opens its own endpoint. Set
``DDSTORE_FORK_WORKERS=1`` for that. On CXI it needs, before libfabric
starts:

- ``FI_CXI_DISABLE_HOST_REGISTER=1`` (always set here unless already set);
- an MR cache monitor that survives fork: with ``DDSTORE_FORK_WORKERS=1``,
  ``FI_MR_CACHE_MONITOR=memhooks`` if it is unset or ``kdreg2`` (Frontier's
  default, which hangs forked workers).

Import pyddstore before mpi4py (MPI starts libfabric), or export them in
the job script.
"""

import os as _os

_os.environ.setdefault("FI_CXI_DISABLE_HOST_REGISTER", "1")
if _os.environ.get("DDSTORE_FORK_WORKERS", "0") not in ("", "0"):
    if _os.environ.get("FI_MR_CACHE_MONITOR", "kdreg2") == "kdreg2":
        _os.environ["FI_MR_CACHE_MONITOR"] = "memhooks"

try:
    from importlib.metadata import version as _version

    __version__ = _version("PyDDStore")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "unknown"

__all__ = ["PyDDStore", "PyDDstoreVarinfo", "FileComm", "HAVE_MPI", "__version__"]


def __getattr__(name):
    if name in ("PyDDStore", "PyDDstoreVarinfo", "FileComm", "HAVE_MPI"):
        from . import _core

        return getattr(_core, name)
    raise AttributeError(f"module 'pyddstore' has no attribute {name!r}")
