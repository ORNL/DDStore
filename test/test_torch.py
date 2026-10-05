"""
pyddstore.torch (DistDataset, DistDatasetReader, ThreadDataLoader) tests —
run with 2+ ranks, e.g.:
  mpirun -n 4 pytest test/test_torch.py -v

Method 0 always; method 1 and the method-2 reader where a CXI device is
present inside a Slurm step (provider: DDSTORE_FABRIC, default cxi).
"""

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from pyddstore.torch import (
    DistDataset,
    DistDatasetReader,
    ThreadDataLoader,
)  # noqa: E402

HAVE_CXI = bool(glob.glob("/dev/cxi*")) and "SLURM_STEP_ID" in os.environ
FABRIC = os.environ.get("DDSTORE_FABRIC", "cxi")
HAVE_GPU = torch.cuda.is_available()
METHODS = [
    0,
    pytest.param(1, marks=pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")),
]
N = 37  # not a multiple of the rank count


class TupleSource(Dataset):
    """Every field kind, values derived from the index."""

    def __len__(self):
        return N

    def __getitem__(self, i):
        return (
            torch.arange(12, dtype=torch.float32).reshape(3, 4)
            + 100 * i,  # torch tensor
            i,  # Python int
            np.full(2, i / 3, dtype=np.float64),  # numpy array
            np.int32(-i),  # numpy scalar
            i % 2 == 0,  # Python bool
            float(i) * 0.5,  # Python float
            torch.tensor([i, i + 1], dtype=torch.uint8),  # small dtype
        )


class DictSource(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return {
            "x": torch.full((2, 3), float(i)),
            "y": torch.tensor(i, dtype=torch.int64),
        }


class SingleSource(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return np.full((5,), i, dtype=np.int32)


def same(a, b):
    """Equal structure, types and values."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and bool(torch.equal(a.cpu(), b.cpu()))
        )
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a == b
    return a == b


def all_ok(comm, ok):
    return comm.allreduce(int(bool(ok)), op=__import__("mpi4py").MPI.LAND)


def make(comm, monkeypatch, source, method, **kw):
    if method != 0:
        monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    return DistDataset(source, f"t{method}", comm, method=method, **kw)


def finish(comm, ds):
    comm.Barrier()  # every rank done reading before any rank tears down
    ds.ddstore.free()


@pytest.mark.parametrize("chunk_size", [None, 1, 4])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("source_cls", [TupleSource, DictSource, SingleSource])
def test_items_match_source(comm, monkeypatch, method, source_cls, chunk_size):
    src = source_cls()
    ds = make(comm, monkeypatch, src, method, chunk_size=chunk_size)
    rng = np.random.default_rng(comm.Get_rank())
    idx = rng.integers(0, N, size=20)
    ok = len(ds) == N
    ok &= all(same(ds[int(i)], src[int(i)]) for i in idx[:5])  # per-sample get()
    batch = ds.__getitems__(idx)  # one get_batch() per field (collective for method 0)
    ok &= all(same(b, src[int(i)]) for b, i in zip(batch, idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_shapes_and_dtypes(comm, monkeypatch):
    ds = make(comm, monkeypatch, DictSource(), 0)
    ok = ds.shapes == {"x": (2, 3), "y": ()} and ds.dtypes == {
        "x": "float32",
        "y": "int64",
    }
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("loader", ["DataLoader", "ThreadDataLoader"])
def test_loaders_match_plain_source(comm, monkeypatch, method, loader):
    """A whole epoch through the loader equals the same loader over the plain
    source (same order, same collation). Collective for method 0: every rank
    iterates the same number of batches."""
    src = TupleSource()
    ds = make(comm, monkeypatch, src, method)
    cls = DataLoader if loader == "DataLoader" else ThreadDataLoader
    kw = {} if cls is DataLoader else {"num_workers": 1 if method == 0 else 2}
    got = list(cls(ds, batch_size=8, **kw))
    ref = list(DataLoader(src, batch_size=8))
    ok = len(got) == len(ref) and all(same(g, r) for g, r in zip(got, ref))
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_per_sample_fallback(comm, monkeypatch):
    monkeypatch.setenv("DDSTORE_BATCH_GET", "0")
    src = TupleSource()
    ds = make(comm, monkeypatch, src, 0)
    idx = list(range(N))[::-3]
    ok = all(same(b, src[i]) for b, i in zip(ds.__getitems__(idx), idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


def test_ddstore_width_groups(comm, monkeypatch):
    if comm.Get_size() < 4:
        pytest.skip("requires at least 4 ranks")
    src = SingleSource()
    ds = make(comm, monkeypatch, src, 0, ddstore_width=2)
    idx = list(range(N))
    ok = all(same(b, src[i]) for b, i in zip(ds.__getitems__(idx), idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


class _Bad(Dataset):
    def __init__(self, kind):
        self.kind = kind

    def __len__(self):
        return N

    def __getitem__(self, i):
        if self.kind == "shape":
            return np.zeros(3 if i % 5 else 4, dtype=np.float32)
        if self.kind == "dtype":
            return torch.zeros(2, dtype=torch.float16)
        if self.kind == "nested":
            return (np.zeros(2), (1, 2))
        return object()


@pytest.mark.parametrize(
    "kind,exc",
    [
        ("shape", ValueError),
        ("dtype", TypeError),
        ("nested", TypeError),
        ("object", TypeError),
    ],
)
@pytest.mark.parametrize("chunk_size", [None, 3])
def test_unsupported_samples_raise(comm, kind, exc, chunk_size):
    with pytest.raises(exc):
        DistDataset(_Bad(kind), "bad", comm, method=0, chunk_size=chunk_size)
    comm.Barrier()


def test_chunk_size_needs_host_storage(comm):
    with pytest.raises(ValueError):
        DistDataset(TupleSource(), "c", comm, method=0, chunk_size=4, add_device="cpu")
    with pytest.raises(ValueError):
        DistDataset(TupleSource(), "c", comm, method=0, chunk_size=0)
    comm.Barrier()


@pytest.mark.skipif(
    not (HAVE_CXI and HAVE_GPU and FABRIC == "cxi"),
    reason="requires the cxi provider and a GPU",
)
def test_gpu_device_and_add_device(comm, monkeypatch):
    src = TupleSource()
    ds = make(comm, monkeypatch, src, 1, device="cuda", add_device="cuda")
    idx = list(range(N))[::-1]
    batch = ds.__getitems__(idx)
    ok = batch[0][0].is_cuda and isinstance(
        batch[0][2], np.ndarray
    )  # tensors on GPU, numpy stays host
    ok &= all(same(b, src[i]) for b, i in zip(batch, idx))
    finish(comm, ds)
    assert all_ok(comm, ok)


@pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")
def test_method2_reader(comm, monkeypatch, tmp_path):
    monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    hs = comm.bcast(str(tmp_path / "hs") if comm.Get_rank() == 0 else None, root=0)
    src = DictSource()
    core = DistDataset(src, "rd", comm, method=2, handshake_dir=hs)
    comm.Barrier()
    ok = True
    if comm.Get_rank() == 0:
        reader = DistDatasetReader("rd", handshake_dir=hs, n_core=comm.Get_size())
        idx = list(range(N))
        ok = len(reader) == N and reader.shapes == core.shapes
        ok &= all(same(b, src[i]) for b, i in zip(reader.__getitems__(idx), idx))
        reader.ddstore.free()
    finish(comm, core)
    assert all_ok(comm, ok)
