"""DDStore: distributed in-memory data store for data-parallel training.

The store itself is ``PyDDStore`` (compiled extension ``pyddstore._core``).
PyTorch helpers (``DistDataset``, ``DistDatasetReader``, ``ThreadDataLoader``)
live in ``pyddstore.torch`` and need PyTorch; importing ``pyddstore`` alone
does not.

The extension (which initializes MPI through mpi4py) is loaded on first use
of ``PyDDStore``, not on ``import pyddstore``: ``from pyddstore.torch import
...`` then imports torch before MPI starts, the order torch + RCCL/NCCL need
to shut down cleanly.
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("PyDDStore")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "unknown"

__all__ = ["PyDDStore", "PyDDstoreVarinfo", "__version__"]


def __getattr__(name):
    if name in ("PyDDStore", "PyDDstoreVarinfo"):
        from . import _core

        return getattr(_core, name)
    raise AttributeError(f"module 'pyddstore' has no attribute {name!r}")
