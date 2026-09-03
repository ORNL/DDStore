"""
GPUDirect RDMA tests (Phase 1: host source -> GPU destination).

Positive path — run with: DDSTORE_FABRIC=cxi mpirun -n 2 pytest test/test_gpu_rdma.py -v
requires a live cxi/Slingshot fabric and at least one visible GPU per rank
(see run-test-gpu.sh). Negative-path tests need neither and always run.
"""

import numpy as np
import pytest
from mpi4py import MPI

import pyddstore as dds

torch = pytest.importorskip("torch")

gpu_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm/HIP GPU"
)


def all_passed(comm, local_ok):
    return comm.allreduce(int(local_ok), op=MPI.LAND)


# ---------------------------------------------------------------------------
# positive path: host source -> poisoned GPU destination, over cxi
# ---------------------------------------------------------------------------


@gpu_required
def test_get_into_gpu_tensor_cxi(comm, monkeypatch):
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4

    store = dds.PyDDStore(comm, method=1)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)  # host source (Phase 1 scope — unchanged)

    store.epoch_begin()
    local_ok = True
    for target_rank in range(size):
        # poison, not zeros: a silent no-op/host-staged-fallback bug would
        # leave this value in place instead of the real remote data.
        out = torch.full((1, ncols), -999.0, dtype=torch.float32, device="cuda")
        store.get("x", out, start=target_rank * nrows)
        expected = float(target_rank + 1)
        ok = bool(torch.all(out.cpu() == expected))
        print(f"[rank {rank}] target_rank={target_rank} expected={expected} "
              f"got={out.cpu().tolist()} ok={ok}", flush=True)
        if not ok:
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


@gpu_required
def test_get_into_gpu_tensor_cxi_large(comm, monkeypatch):
    """Diagnostic: same as test_get_into_gpu_tensor_cxi but with a transfer
    well over FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD (default 4096 bytes), to
    check whether CXI's small-transfer 'safe load/store' HMEM path is what's
    silently no-op'ing, vs. the registration approach itself being broken.
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4096  # 4096 floats/row = 16384 bytes >> 4096-byte threshold

    store = dds.PyDDStore(comm, method=1)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    local_ok = True
    for target_rank in range(size):
        out = torch.full((1, ncols), -999.0, dtype=torch.float32, device="cuda")
        store.get("x", out, start=target_rank * nrows)
        expected = float(target_rank + 1)
        ok = bool(torch.all(out.cpu() == expected))
        nonpoison = int((out.cpu() != -999.0).sum())
        print(f"[rank {rank}] target_rank={target_rank} expected={expected} "
              f"ok={ok} nonpoison_count={nonpoison}/{ncols} "
              f"sample={out.cpu().flatten()[:8].tolist()}", flush=True)
        if not ok:
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


@gpu_required
def test_get_host_to_host_cxi(comm, monkeypatch):
    """Diagnostic: same transfer as above, but into a host numpy buffer
    (bypasses HMEM entirely) -- isolates whether plain host-to-host RDMA
    over cxi works correctly in this environment, independent of GPU support.
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4

    store = dds.PyDDStore(comm, method=1)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    local_ok = True
    for target_rank in range(size):
        out = np.full((1, ncols), -999.0, dtype=np.float32)
        store.get("x", out, start=target_rank * nrows)
        expected = float(target_rank + 1)
        ok = bool(np.all(out == expected))
        print(f"[rank {rank}] target_rank={target_rank} expected={expected} "
              f"got={out.tolist()} ok={ok}", flush=True)
        if not ok:
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


# ---------------------------------------------------------------------------
# negative paths: clear, early errors -- no live cxi fabric required
# ---------------------------------------------------------------------------


@gpu_required
def test_gpu_buffer_rejected_on_hsn(comm, monkeypatch):
    # Set up with cxi so that add() and init_fabric succeed (hsn is not
    # available on all machines, e.g. Perlmutter which is CXI-only).
    # Then switch DDSTORE_FABRIC to hsn before get() — the Python-level check
    # in pyddstore.pyx reads the env var at get() time and rejects GPU buffers
    # with a clear error before touching the fabric.
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    store = dds.PyDDStore(comm, method=1)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)

    monkeypatch.setenv("DDSTORE_FABRIC", "hsn")
    out = torch.zeros((1, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="cxi"):
        store.get("x", out, start=0)
    store.free()


@gpu_required
def test_gpu_buffer_rejected_on_method0(comm):
    store = dds.PyDDStore(comm, method=0)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)

    out = torch.zeros((1, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="method"):
        store.get("x", out, start=0)
    store.free()


@gpu_required
def test_gpu_source_rejected_by_add(comm):
    store = dds.PyDDStore(comm, method=1)
    data = torch.ones((4, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(NotImplementedError):
        store.add("x", data)
