# distutils: language=c++
# cython: language_level=3
# cython: language=c++

import os
import time

import mpi4py.MPI as MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

import cpu_nic_map

import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp.typeinfo cimport type_info

from cpython.version cimport PY_MAJOR_VERSION

cpdef str b2s(bytes x):
    if PY_MAJOR_VERSION < 3:
        return str(x)
    else:
        return x.decode()

cpdef bytes s2b(str x):
    if PY_MAJOR_VERSION < 3:
        return <bytes>x
    else:
        return x.encode()

def _is_cuda_tensor(obj):
    """True if obj is a torch.Tensor on a CUDA/HIP device.

    Torch is optional — imported lazily so DDStore2 has no hard dependency
    on it. If unimportable, no object is ever considered a CUDA tensor.
    """
    try:
        import torch
    except ImportError:
        return False
    return isinstance(obj, torch.Tensor) and obj.is_cuda

# Mirrors libfabric's enum fi_hmem_iface (rdma/fi_domain.h): FI_HMEM_SYSTEM=0,
# FI_HMEM_CUDA=1, FI_HMEM_ROCR=2. Kept as plain ints here (rather than
# cimporting the C enum) since only these two values are ever produced by
# _hmem_iface_for() below -- torch itself is either a CUDA or a ROCm build,
# never both.
_FI_HMEM_CUDA = 1
_FI_HMEM_ROCR = 2

def _hmem_iface_for(tensor):
    """fi_hmem_iface value for a CUDA tensor: ROCr on a ROCm/HIP build of
    torch (AMD), CUDA otherwise (NVIDIA). Only call when _is_cuda_tensor()
    is already True.
    """
    import torch
    return _FI_HMEM_ROCR if torch.version.hip is not None else _FI_HMEM_CUDA


def _check_gpu_fabric_preconditions(int method, str what):
    """Shared method=1/2 + DDSTORE_FABRIC=cxi precondition check for a GPU
    (CUDA/HIP) buffer passed to add() or get(). `what` customizes the error
    wording ("GPU source buffer" / "GPU destination buffer").
    """
    if method not in (1, 2):
        raise RuntimeError(
            "%s requires method=1 or 2 (libfabric), got method=%d" % (what, method))
    provider = os.environ.get("DDSTORE_FABRIC", "hsn")
    if provider != "cxi":
        raise RuntimeError(
            "%s requires DDSTORE_FABRIC=cxi (current DDSTORE_FABRIC=%r); "
            "the hsn (tcp;ofi_rxm) path does not support FI_HMEM. Set "
            "DDSTORE_FABRIC=cxi or pass a host (CPU) numpy array instead."
            % (what, provider))

def _check_dtype(arr, bint is_gpu):
    """Raise NotImplementedError unless arr's dtype is one DDStore supports.
    add()/get() dispatch on item size alone (1/4/8 bytes), so this is what
    keeps e.g. float16 or complex64 from slipping through on a size match.
    """
    if is_gpu:
        import torch
        ok = arr.dtype in (torch.int32, torch.int64, torch.uint8,
                           torch.float32, torch.float64, torch.bool)
    else:
        ok = arr.dtype in (np.int32, np.int64, np.uint8,
                           np.float32, np.float64, np.bool_)
    if not ok:
        raise NotImplementedError("unsupported dtype: %s" % arr.dtype)

cdef extern from "ddstore.hpp":
    ctypedef struct VarInfo:
        string name
        int disp
        int itemsize


    cdef cppclass DDStore:
        DDStore()
        DDStore(libmpi.MPI_Comm comm)
        DDStore(int method, libmpi.MPI_Comm comm)
        # Method 2: core member (with MPI communicator; n_core == comm size)
        DDStore(int method, libmpi.MPI_Comm comm,
                string handshake_dir)
        # Method 2: extra member (no MPI communicator)
        DDStore(int method, string handshake_dir, int n_core)
        void add[T](string name, T* buffer, long nrows, int disp, int hmem_iface) except +
        void get[T](string name, long start, long count, T* buffer, int hmem_iface) except + nogil
        void get_batch[T](string name, const long *idx, long n, T* buffer, int hmem_iface) except + nogil
        void epoch_begin()
        void epoch_end()
        void free()
        void init(string name, long nrows, int disp, int itemsize) except +
        void update[T](string name, T* buffer, long nrows, long offset) except +
        void join(string name) except +
        void query(string name, VarInfo &varinfo) except +
        long size(string name) except +
        void profile(string name, unsigned long long *out) except +

cdef class PyDDstoreVarinfo:
    cdef VarInfo c_varinfo

    def __cinit__(self):
        pass

cdef class PyDDStore:
    cdef DDStore *c_ddstore
    cdef int method
    # Keepalive for GPU tensors passed to add(): C++ holds a raw pointer
    # into them with no copy and no refcounting (see ddstore.hpp add()'s
    # lifetime-contract comment) -- this dict keeps the Python reference
    # alive for as long as the variable stays registered.
    cdef dict _gpu_owned_buffers
    # DDSTORE_PROFILE=1: Python-side get() timing (see get_profile()).
    cdef bint _prof
    cdef double _prof_get_s
    cdef double _prof_sync_s
    cdef long _prof_gets

    def __cinit__(self, comm_or_none=None, int method=0,
                  str handshake_dir="", int n_core=0, nic_map=None):
        """
        Constructors:
          PyDDStore(comm)                          — method 0, MPI
          PyDDStore(comm, method=1)                — method 1, libfabric+MPI
          PyDDStore(comm, method=2,                — method 2, core member
                    handshake_dir="/path")            (n_core == comm size)
          PyDDStore(None, method=2,                — method 2, extra member
                    handshake_dir="/path", n_core=N)

        nic_map: optional precomputed CPU->NIC map string (see
          cpu_nic_map.py --env) used to select FABRIC_IFACE for this rank's
          CPU affinity, for method=1/2. Takes priority over the
          DDSTORE_NIC_MAP env var. Only used if FABRIC_IFACE isn't already
          set in the environment.
        """
        cdef MPI.Comm mpi_comm
        self.method = method
        self._gpu_owned_buffers = {}
        self._prof = os.environ.get("DDSTORE_PROFILE", "0") not in ("", "0")
        self._prof_get_s = 0.0
        self._prof_sync_s = 0.0
        self._prof_gets = 0
        if method != 0:
            cpu_nic_map.select_fabric_iface(nic_map=nic_map)
        if method == 2:
            if not handshake_dir:
                raise ValueError(
                    "method=2 requires handshake_dir (got handshake_dir=%r)"
                    % handshake_dir)
            if comm_or_none is None:
                # Extra member: no MPI communicator, n_core must be given
                if n_core <= 0:
                    raise ValueError(
                        "method=2 extra member requires n_core > 0 "
                        "(got n_core=%d)" % n_core)
                self.c_ddstore = new DDStore(method,
                                             s2b(handshake_dir), n_core)
            else:
                # Core member with file-based handshake; n_core is derived
                # from the communicator size.
                mpi_comm = comm_or_none
                self.c_ddstore = new DDStore(method, mpi_comm.ob_mpi,
                                             s2b(handshake_dir))
        else:
            # Methods 0 and 1: standard MPI constructor
            if comm_or_none is None:
                raise ValueError(
                    "method=%d requires a valid MPI communicator "
                    "(got comm_or_none=None)" % method)
            mpi_comm = comm_or_none
            self.c_ddstore = new DDStore(method, mpi_comm.ob_mpi)

    def __dealloc__(self):
        if self.c_ddstore != NULL:
            del self.c_ddstore
            self.c_ddstore = NULL
        self._gpu_owned_buffers.clear()

    def add(self, str name, arr):
        cdef size_t ptr
        cdef int itemsize
        cdef int iface
        cdef long nrows = arr.shape[0]
        cdef int disp
        cdef bint is_gpu = _is_cuda_tensor(arr)
        _check_dtype(arr, is_gpu)
        if is_gpu:
            _check_gpu_fabric_preconditions(self.method, "GPU source buffer")
            assert arr.is_contiguous()
            if name in self._gpu_owned_buffers:
                raise RuntimeError(
                    "add() called again for variable '%s' with a GPU source "
                    "buffer; re-adding an existing variable name is not "
                    "supported (the original registration would remain "
                    "active in C++ while its Python keepalive reference is "
                    "replaced here, risking a dangling pointer)" % name)
            import torch
            # Flush any pending/async GPU compute-kernel writes to `arr`
            # before handing it to RDMA -- otherwise the transfer can be
            # silently masked by stale GPU cache content from a preceding,
            # not-yet-retired write to the same memory.
            torch.cuda.synchronize(device=arr.device)
            ptr = arr.data_ptr()
            itemsize = arr.element_size()
            disp = arr.numel() // nrows
            iface = _hmem_iface_for(arr)
        else:
            assert arr.flags.c_contiguous
            ptr = arr.ctypes.data
            itemsize = arr.itemsize
            disp = arr.size // nrows
            iface = 0

        # DDStore::add<T>() only uses T through sizeof(T), so dispatching on
        # item size is enough.
        cdef string cname = s2b(name)
        if itemsize == 1:
            self.c_ddstore.add(cname, <char *> ptr, nrows, disp, iface)
        elif itemsize == 4:
            self.c_ddstore.add(cname, <int *> ptr, nrows, disp, iface)
        else:
            self.c_ddstore.add(cname, <long *> ptr, nrows, disp, iface)

        if is_gpu:
            # Keepalive: DDStore now holds a raw pointer into arr's storage
            # with no copy and no C++-level refcounting -- see ddstore.hpp
            # add()'s lifetime-contract doc comment. Must outlive this
            # variable's registration; cleared in free()/__dealloc__.
            self._gpu_owned_buffers[name] = arr

    def get(self, str name, arr, long start=0):
        cdef double t_get = time.perf_counter() if self._prof else 0.0
        cdef double t_sync
        cdef long count = arr.shape[0]
        cdef size_t ptr
        cdef int itemsize
        cdef int iface
        cdef bint is_gpu = _is_cuda_tensor(arr)
        _check_dtype(arr, is_gpu)
        if is_gpu:
            _check_gpu_fabric_preconditions(self.method, "GPU destination buffer")
            assert arr.is_contiguous()
            import torch
            # See the matching comment in add() for what this guards against.
            if self._prof:
                t_sync = time.perf_counter()
                torch.cuda.synchronize(device=arr.device)
                self._prof_sync_s += time.perf_counter() - t_sync
            else:
                torch.cuda.synchronize(device=arr.device)
            ptr = arr.data_ptr()
            itemsize = arr.element_size()
            iface = _hmem_iface_for(arr)
        else:
            assert arr.flags.c_contiguous
            ptr = arr.ctypes.data
            itemsize = arr.itemsize
            iface = 0

        # DDStore::get<T>() only uses T for its sizeof(T) == itemsize check
        # and the pointer cast -- the transfer itself is a byte copy -- so
        # dispatching on item size is enough. The read runs without the GIL,
        # so other Python threads (e.g. a training loop while a background
        # thread prefetches) keep running. Method 1/2 reads make no MPI calls.
        cdef string cname = s2b(name)
        with nogil:
            if itemsize == 1:
                self.c_ddstore.get(cname, start, count, <char *> ptr, iface)
            elif itemsize == 4:
                self.c_ddstore.get(cname, start, count, <int *> ptr, iface)
            else:
                self.c_ddstore.get(cname, start, count, <long *> ptr, iface)
        if self._prof:
            self._prof_get_s += time.perf_counter() - t_get
            self._prof_gets += 1

    def get_batch(self, str name, arr, indices):
        """Read rows `indices` (global row ids, any order, repeats allowed)
        into `arr`, whose first dimension must equal len(indices): row i of
        `arr` receives row indices[i]. Same buffer rules as get(); for
        method 1/2 all reads of the batch are in flight together, under one
        lock acquisition and (GPU destination) one device sync."""
        cdef double t_get = time.perf_counter() if self._prof else 0.0
        cdef double t_sync
        cdef np.ndarray idx = np.ascontiguousarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        cdef long n = idx.shape[0]
        if arr.shape[0] != n:
            raise ValueError(
                "arr has %d rows but %d indices were given" % (arr.shape[0], n))
        cdef const long *cidx = <const long *> idx.data
        cdef size_t ptr
        cdef int itemsize
        cdef int iface
        cdef bint is_gpu = _is_cuda_tensor(arr)
        _check_dtype(arr, is_gpu)
        if is_gpu:
            _check_gpu_fabric_preconditions(self.method, "GPU destination buffer")
            assert arr.is_contiguous()
            import torch
            # Same reason as get(), once per batch.
            if self._prof:
                t_sync = time.perf_counter()
                torch.cuda.synchronize(device=arr.device)
                self._prof_sync_s += time.perf_counter() - t_sync
            else:
                torch.cuda.synchronize(device=arr.device)
            ptr = arr.data_ptr()
            itemsize = arr.element_size()
            iface = _hmem_iface_for(arr)
        else:
            assert arr.flags.c_contiguous
            ptr = arr.ctypes.data
            itemsize = arr.itemsize
            iface = 0

        cdef string cname = s2b(name)
        with nogil:
            if itemsize == 1:
                self.c_ddstore.get_batch(cname, cidx, n, <char *> ptr, iface)
            elif itemsize == 4:
                self.c_ddstore.get_batch(cname, cidx, n, <int *> ptr, iface)
            else:
                self.c_ddstore.get_batch(cname, cidx, n, <long *> ptr, iface)
        if self._prof:
            self._prof_get_s += time.perf_counter() - t_get
            self._prof_gets += 1

    def get_profile(self, str name):
        """DDSTORE_PROFILE=1 timing for `name` (methods 1/2), in seconds.

        C++ counters for this variable: calls (get + get_batch), rows,
        lock_wait, mr (recv-MR cache check/registration), mr_miss
        (re-registrations), read (posting fi_read), cq (waiting for
        completion). Python counters for this
        store, across all variables: py_gets, py_get (whole get() calls),
        py_sync (torch.cuda.synchronize on the GPU-destination path).
        """
        cdef unsigned long long c[7]
        self.c_ddstore.profile(s2b(name), c)
        return {
            "calls": c[0], "lock_wait": c[1] * 1e-9, "mr": c[2] * 1e-9,
            "mr_miss": c[3], "read": c[4] * 1e-9, "cq": c[5] * 1e-9,
            "rows": c[6],
            "py_gets": self._prof_gets, "py_get": self._prof_get_s,
            "py_sync": self._prof_sync_s,
        }

    def epoch_begin(self):
        self.c_ddstore.epoch_begin()

    def epoch_end(self):
        self.c_ddstore.epoch_end()

    def free(self):
        self.c_ddstore.free()
        self._gpu_owned_buffers.clear()

    def init(self, str name, long nrows, int disp, int itemsize=1):
        self.c_ddstore.init(s2b(name), nrows, disp, itemsize)

    def update(self, str name, arr, long offset):
        if _is_cuda_tensor(arr):
            raise NotImplementedError(
                "update() only supports host (numpy) buffers -- the "
                "init()/update() path is host-only; pass arr.cpu().numpy() "
                "instead, or add() the GPU tensor directly")
        cdef np.ndarray np_arr = arr
        assert np_arr.flags.c_contiguous
        cdef long nrows = np_arr.shape[0]
        if np_arr.dtype == np.int32:
            self.c_ddstore.update(s2b(name), <int *> np_arr.data, nrows, offset)
        elif np_arr.dtype == np.int64:
            self.c_ddstore.update(s2b(name), <long *> np_arr.data, nrows, offset)
        elif np_arr.dtype == np.uint8:
            self.c_ddstore.update(s2b(name), <char *> np_arr.data, nrows, offset)
        elif np_arr.dtype == np.float32:
            self.c_ddstore.update(s2b(name), <float *> np_arr.data, nrows, offset)
        elif np_arr.dtype == np.float64:
            self.c_ddstore.update(s2b(name), <double *> np_arr.data, nrows, offset)
        elif np_arr.dtype == np.bool_:
            self.c_ddstore.update(s2b(name), <char *> np_arr.data, nrows, offset)
        else:
            raise NotImplementedError

    def join(self, str name):
        """Method 2 extra member: discover variable published by core members."""
        self.c_ddstore.join(s2b(name))

    def info(self, str name):
        """Return (total_rows, disp, itemsize) for an added or joined variable."""
        cdef VarInfo vi
        self.c_ddstore.query(s2b(name), vi)
        total_rows = self.c_ddstore.size(s2b(name))
        return (total_rows, vi.disp, vi.itemsize)
