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
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("PyDDStore")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "unknown"

__all__ = ["PyDDStore", "PyDDstoreVarinfo", "HAVE_MPI", "__version__"]


def __getattr__(name):
    if name in ("PyDDStore", "PyDDstoreVarinfo", "HAVE_MPI"):
        from . import _core

        return getattr(_core, name)
    raise AttributeError(f"module 'pyddstore' has no attribute {name!r}")
