"""
GPUDirect RDMA tests: GPU destination (Phase 1), GPU source (Phase 2), both,
negative paths, and concurrent get() from multiple threads.

Positive path — run with: DDSTORE_FABRIC=cxi mpirun -n 2 pytest test/test_gpu_rdma.py -v
requires a live cxi/Slingshot fabric and at least one visible GPU per rank.
Negative-path tests need neither and always run.
"""

import threading

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
        print(
            f"[rank {rank}] target_rank={target_rank} expected={expected} "
            f"got={out.cpu().tolist()} ok={ok}",
            flush=True,
        )
        if not ok:
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


@gpu_required
def test_get_into_gpu_tensor_cxi_compute_kernel_read(comm, monkeypatch):
    """Diagnostic: does reading the RDMA destination via a GPU COMPUTE
    KERNEL (not a .cpu() DMA-engine copy) crash/fault, unlike every other
    test in this file which always reads back via .cpu()? A real training
    loop (examples/vae/vae-ddp.py --gpu-dest) feeds the destination tensor
    directly into model(data) -- a compute-kernel read -- and hit
    HSA_STATUS_ERROR_EXCEPTION hardware faults at real batch-loop scale,
    something none of the .cpu()-based tests in this file have ever
    reproduced. Hypothesis: the NIC's P2P write into GPU memory isn't
    visible/coherent to compute cores (cache/TLB gap) the way it is to the
    DMA engine .cpu() uses -- untested by every other test here. Also loops
    many iterations back-to-back (no pauses) to mirror DataLoader's rapid
    per-sample get() calls, in case repeated register/deregister at a
    reused address (PyTorch's allocator likely returns the same block each
    time for same-shape torch.empty() in a tight loop) is a contributing
    factor rather than the compute-kernel read alone.
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4
    n_iters = 200

    store = dds.PyDDStore(comm, method=1)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    target_rank = (rank + 1) % size
    expected = float(target_rank + 1)
    for i in range(n_iters):
        # torch.empty (not full/poisoned): mirrors DistDataset.get()'s real
        # allocation exactly, and a fresh, uninitialized block is what a
        # compute kernel would actually read if the transfer no-op'd --
        # closer to the real crash scenario than a poisoned buffer.
        out = torch.empty((1, ncols), dtype=torch.float32, device="cuda")
        store.get("x", out, start=target_rank * nrows)
        # GPU compute-kernel read (not .cpu()): elementwise op launches a
        # real kernel touching `out`'s memory from the compute cores.
        diff = (out - expected).abs().sum()
        # Forces the host to wait for the kernel and surfaces any async
        # HIP error at this point (torch raises a RuntimeError mentioning
        # the HIP error, or the process aborts, same as the real crash).
        torch.cuda.synchronize()
        if i % 50 == 0:
            print(f"[rank {rank}] iter={i} diff={diff.item()}", flush=True)
    store.epoch_end()
    print(
        f"[rank {rank}] completed {n_iters} iterations without a HIP error", flush=True
    )
    # Wait for every rank's reads before tearing down: otherwise a fast rank
    # can free its endpoint while a peer is still reading (PTLTE_NOT_FOUND).
    comm.Barrier()
    store.free()


@gpu_required
def test_get_into_gpu_tensor_cxi_matrix(comm, monkeypatch):
    """Isolates which factor actually determines pass/fail: buffer
    allocation method (torch.empty, uninitialized vs torch.full, poisoned
    via a GPU compute-kernel write) crossed with readback method (.cpu()
    DMA copy vs GPU compute-kernel read + torch.cuda.synchronize()).
    Written when test_get_into_gpu_tensor_cxi (poison + .cpu()) reliably
    failed on Frontier while test_get_into_gpu_tensor_cxi_compute_kernel_read
    (empty + compute-kernel read) passed; runs all 4 combinations to show
    which axis (allocation vs readback) matters. Both now pass, since
    PyDDStore.get() synchronizes the device before every GPU-destination
    transfer (see test_get_into_gpu_tensor_cxi_sync_before_get).
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4
    POISON = -999.0

    store = dds.PyDDStore(comm, method=1)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)
    store.epoch_begin()

    target_rank = (rank + 1) % size
    expected = float(target_rank + 1)
    results = {}
    for alloc in ("empty", "poison"):
        for readback in ("cpu", "kernel"):
            if alloc == "empty":
                out = torch.empty((1, ncols), dtype=torch.float32, device="cuda")
            else:
                out = torch.full((1, ncols), POISON, dtype=torch.float32, device="cuda")
            store.get("x", out, start=target_rank * nrows)
            if readback == "cpu":
                snapshot = out.cpu()
                ok = bool(torch.all(snapshot == expected))
                detail = snapshot.tolist()
            else:
                diff = (out - expected).abs().sum()
                torch.cuda.synchronize()
                ok = bool(diff.item() == 0.0)
                detail = f"diff={diff.item()}"
            key = f"alloc={alloc},readback={readback}"
            results[key] = ok
            print(f"[rank {rank}] {key} ok={ok} detail={detail}", flush=True)

    store.epoch_end()
    gathered = comm.gather(results, root=0)
    if rank == 0:
        print(f"[rank 0] ALL RESULTS: {gathered}", flush=True)
    store.free()
    # Fail loudly with the full matrix visible in the log even if only one
    # combination is wrong -- this test is diagnostic, not a pass/fail gate.
    assert all(results.values()), f"[rank {rank}] matrix results: {results}"


@gpu_required
def test_get_into_gpu_tensor_cxi_sync_before_get(comm, monkeypatch):
    """Follow-up to test_get_into_gpu_tensor_cxi_matrix's finding: RDMA into
    a torch.full()-poisoned (GPU-kernel-written) buffer fails, but into a
    torch.empty() (untouched) buffer works -- readback method is
    irrelevant. Hypothesis: a cache-coherency gap where the NIC's RDMA
    write doesn't invalidate whatever the GPU cache still holds from the
    prior compute-kernel write. Tests whether an explicit
    torch.cuda.synchronize() between the poisoning kernel and the RDMA
    get() call (forcing the kernel write to fully retire/flush first) is
    enough to fix it.
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4
    POISON = -999.0

    store = dds.PyDDStore(comm, method=1)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)
    store.epoch_begin()

    target_rank = (rank + 1) % size
    expected = float(target_rank + 1)

    out = torch.full((1, ncols), POISON, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()  # <-- the fix under test: flush the poison write first
    store.get("x", out, start=target_rank * nrows)
    snapshot = out.cpu()
    ok = bool(torch.all(snapshot == expected))
    print(f"[rank {rank}] sync-before-get: ok={ok} got={snapshot.tolist()}", flush=True)

    store.epoch_end()
    store.free()
    assert all_passed(comm, ok)


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
        print(
            f"[rank {rank}] target_rank={target_rank} expected={expected} "
            f"ok={ok} nonpoison_count={nonpoison}/{ncols} "
            f"sample={out.cpu().flatten()[:8].tolist()}",
            flush=True,
        )
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
        print(
            f"[rank {rank}] target_rank={target_rank} expected={expected} "
            f"got={out.tolist()} ok={ok}",
            flush=True,
        )
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
def test_gpu_source_rejected_on_method0(comm):
    store = dds.PyDDStore(comm, method=0)
    data = torch.ones((4, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="method"):
        store.add("x", data)


@gpu_required
def test_gpu_source_rejected_on_hsn(comm, monkeypatch):
    monkeypatch.setenv("DDSTORE_FABRIC", "hsn")
    store = dds.PyDDStore(comm, method=1)
    data = torch.ones((4, 4), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="cxi"):
        store.add("x", data)


# ---------------------------------------------------------------------------
# Phase 2: GPU-resident producer (add()) -- host/GPU destination, over cxi
# ---------------------------------------------------------------------------
#
# These pass on Frontier (ROCm + cxi).


@gpu_required
def test_add_from_gpu_tensor_host_dest_cxi(comm, monkeypatch):
    """GPU source -> host (poisoned) destination. Isolates that the SEND
    side specifically works, independent of Phase 1's already-proven
    receive side.
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4

    store = dds.PyDDStore(comm, method=1)
    data = torch.full(
        (nrows, ncols), float(rank + 1), dtype=torch.float32, device="cuda"
    )
    store.add("x", data)  # GPU source -- Phase 2

    store.epoch_begin()
    local_ok = True
    for target_rank in range(size):
        out = np.full((1, ncols), -999.0, dtype=np.float32)
        store.get("x", out, start=target_rank * nrows)
        ok = bool(np.all(out == float(target_rank + 1)))
        if not ok:
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


@gpu_required
def test_add_from_gpu_tensor_gpu_dest_cxi(comm, monkeypatch):
    """Full Phase 2 scenario: both ends device memory, method=1."""
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4

    store = dds.PyDDStore(comm, method=1)
    data = torch.full(
        (nrows, ncols), float(rank + 1), dtype=torch.float32, device="cuda"
    )
    store.add("x", data)

    store.epoch_begin()
    local_ok = True
    for target_rank in range(size):
        out = torch.full((1, ncols), -999.0, dtype=torch.float32, device="cuda")
        store.get("x", out, start=target_rank * nrows)
        expected = float(target_rank + 1)
        ok = bool(torch.all(out.cpu() == expected))
        if not ok:
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


@gpu_required
def test_add_from_gpu_tensor_gpu_dest_cxi_method2(comm, monkeypatch, tmp_path):
    """Same as test_add_from_gpu_tensor_gpu_dest_cxi but method=2
    (file-based handshake, core+extra split) -- the transport
    examples/vae/distdataset.py's DistDatasetReader actually uses. Includes
    a self-read check (core rank both add()s and get()s its own data,
    mirroring test_method2_core.py's self-check pattern) to verify the
    independent send_hmem_iface/mr vs recv_hmem_iface/recv_mr fields don't
    interfere with each other.
    """
    monkeypatch.setenv("DDSTORE_FABRIC", "cxi")
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks for a genuine remote read")
    nrows, ncols = 8, 4

    hs_dir = comm.bcast(
        str(tmp_path / "ddstore_hs_add_method2") if rank == 0 else None, root=0
    )

    core_store = dds.PyDDStore(comm, method=2, handshake_dir=hs_dir)
    data = torch.full(
        (nrows, ncols), float(rank + 1), dtype=torch.float32, device="cuda"
    )
    core_store.add("x", data)  # GPU source -- Phase 2
    comm.Barrier()

    # Self-read: every core rank reads its own just-added shard back.
    local_ok = True
    out_self = torch.full((1, ncols), -999.0, dtype=torch.float32, device="cuda")
    core_store.get("x", out_self, start=rank * nrows)
    if not bool(torch.all(out_self.cpu() == float(rank + 1))):
        local_ok = False

    # Extra member: a separate instance joins and reads every rank's shard.
    if rank == 0:
        extra_store = dds.PyDDStore(None, method=2, handshake_dir=hs_dir, n_core=size)
        extra_store.join("x")
        for target_rank in range(size):
            out = torch.full((1, ncols), -999.0, dtype=torch.float32, device="cuda")
            extra_store.get("x", out, start=target_rank * nrows)
            expected = float(target_rank + 1)
            if not bool(torch.all(out.cpu() == expected)):
                local_ok = False
        extra_store.free()

    comm.Barrier()
    assert all_passed(comm, local_ok)
    core_store.free()


# ---------------------------------------------------------------------------
# thread-safety: concurrent get() calls from multiple Python threads
# ---------------------------------------------------------------------------


def test_concurrent_get_thread_safety(comm, monkeypatch):
    """DDStore::get() releases the GIL for its blocking transfer (see the
    `with nogil:` block in pyddstore.pyx), so multiple Python threads can
    genuinely be inside DDStore::get() at the same time. Without
    synchronization, concurrent calls on the same variable would race on
    the CQ poll loop and the recv-MR region cache in common.cxx (confirmed
    by direct experiment: disabling the protection crashed with "double
    free or corruption").

    That protection now lives inside DDStore itself -- a per-variable
    `pthread_mutex_t` on `struct fabric_state` (include/common.h), taken
    via the `fabric_state_lock_guard` RAII helper around get()'s critical
    section in include/ddstore.hpp. This test calls PyDDStore.get()
    directly from multiple threads with **no lock at the Python level at
    all** -- it would catch a regression if
    that C++-level protection were ever removed or narrowed.
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
    comm.Barrier()

    store.epoch_begin()
    results = {}
    errors = []

    def unlocked_get(target_rank):
        out = np.full((1, ncols), -999.0, dtype=np.float32)
        store.get("x", out, start=target_rank * nrows)
        results[target_rank] = out.copy()

    def worker(target_ranks):
        try:
            for target_rank in target_ranks:
                unlocked_get(target_rank)
        except Exception as exc:  # noqa: BLE001 - surface any thread exception
            errors.append(exc)

    n_threads = 4
    threads = [
        threading.Thread(target=worker, args=(list(range(t, size, n_threads)),))
        for t in range(min(n_threads, size))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.epoch_end()

    assert not errors, f"worker thread(s) raised: {errors}"
    local_ok = True
    for target_rank in range(size):
        expected = float(target_rank + 1)
        got = results[target_rank]
        ok = bool(np.all(got == expected))
        if not ok:
            local_ok = False
        print(
            f"[rank {rank}] target_rank={target_rank} expected={expected} "
            f"got={got.tolist()} ok={ok}",
            flush=True,
        )

    assert all_passed(comm, local_ok)
    store.free()
