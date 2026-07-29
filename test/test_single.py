"""
Single-rank tests — run with: mpirun -n 1 pytest test/test_single.py -v

No remote memory access; all data is local.
"""
import numpy as np
import pytest
from mpi4py import MPI

import pyddstore as dds


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_store(comm, method=0):
    return dds.PyDDStore(comm, method=method)


def make_data(dtype, shape=(4, 4)):
    nrows, ncols = shape
    n = nrows * ncols
    if dtype == np.bool_:
        return np.array([i % 2 == 0 for i in range(n)], dtype=dtype).reshape(shape)
    return np.arange(n, dtype=dtype).reshape(shape)


def roundtrip(store, name, data):
    """add data, get every row back, return stacked result."""
    store.add(name, data)
    nrows, ncols = data.shape
    out = np.zeros((1, ncols), dtype=data.dtype)
    results = []
    store.epoch_begin()
    for i in range(nrows):
        store.get(name, out, start=i)
        results.append(out.copy())
    store.epoch_end()
    return np.concatenate(results)


# ---------------------------------------------------------------------------
# dtype coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [
    np.int32, np.int64, np.uint8, np.float32, np.float64, np.bool_,
])
def test_add_get_dtypes(comm, dtype):
    store = make_store(comm)
    data = make_data(dtype)
    result = roundtrip(store, "x", data)
    np.testing.assert_array_equal(result, data)
    store.free()


# ---------------------------------------------------------------------------
# init / update / get
# ---------------------------------------------------------------------------

def test_init_update_get(comm):
    store = make_store(comm)
    nrows, ncols = 4, 8
    store.init("y", nrows, ncols, itemsize=4)  # float32

    data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
    store.update("y", data, offset=0)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    store.get("y", out, start=2)
    store.epoch_end()

    np.testing.assert_array_equal(out[0], data[2])
    store.free()


def test_update_partial(comm):
    store = make_store(comm)
    nrows, ncols = 8, 4
    store.init("z", nrows, ncols, itemsize=4)

    first_half = np.ones((4, ncols), dtype=np.float32)
    second_half = np.full((4, ncols), 2.0, dtype=np.float32)
    store.update("z", first_half, offset=0)
    store.update("z", second_half, offset=4)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    store.get("z", out, start=0)
    assert out[0, 0] == 1.0
    store.get("z", out, start=4)
    assert out[0, 0] == 2.0
    store.epoch_end()
    store.free()


# ---------------------------------------------------------------------------
# multiple variables
# ---------------------------------------------------------------------------

def test_multiple_variables(comm):
    store = make_store(comm)
    a = np.ones((4, 4), dtype=np.float32)
    b = np.full((4, 4), 2.0, dtype=np.float32)
    store.add("a", a)
    store.add("b", b)

    store.epoch_begin()
    out = np.zeros((1, 4), dtype=np.float32)
    store.get("a", out, start=0)
    assert out[0, 0] == 1.0
    store.get("b", out, start=0)
    assert out[0, 0] == 2.0
    store.epoch_end()
    store.free()


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------

def test_get_out_of_range(comm):
    store = make_store(comm)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(Exception):
        store.get("x", out, start=100)
    store.epoch_end()
    store.free()


def test_get_negative_index(comm):
    store = make_store(comm)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(Exception):
        store.get("x", out, start=-1)
    store.epoch_end()
    store.free()


def test_get_unknown_name(comm):
    store = make_store(comm)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(Exception):
        store.get("missing", out, start=0)
    store.epoch_end()
    store.free()


def test_get_wrong_dtype(comm):
    store = make_store(comm)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, 4), dtype=np.float64)  # wrong dtype
    with pytest.raises(Exception):
        store.get("x", out, start=0)
    store.epoch_end()
    store.free()


# ---------------------------------------------------------------------------
# double free safety
# ---------------------------------------------------------------------------

def test_double_free(comm):
    store = make_store(comm)
    data = np.ones((4, 4), dtype=np.float32)
    store.add("x", data)
    store.free()
    store.free()  # must not crash
