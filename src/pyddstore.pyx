# distutils: language=c++
# cython: language_level=3
# cython: language=c++

import os

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

def _should_sync_before_rdma(tensor):
    """Whether to torch.cuda.synchronize() a GPU buffer before handing it
    to RDMA (see the sync call sites in add()/get() for what this guards
    against). Controlled by DDSTORE_GPU_SYNC (case-insensitive):
      "always" -- always sync.
      "never"  -- never sync (only override this if you've independently
                  confirmed your workload is safe without it -- see below).
      "auto" (default, or any other/unset value) -- sync on ROCm/HIP
                  builds of torch, skip on CUDA builds.

    Rationale for the auto default: confirmed by direct testing that
    ROCm's HMEM-over-CXI path needs this sync -- without it, a GPU
    buffer's RDMA transfer can be silently masked by stale GPU cache
    content from a preceding, not-yet-retired compute-kernel write to the
    same memory (observed as both silent data corruption in small tests
    and HSA_STATUS_ERROR_EXCEPTION hardware faults in a real, sustained
    training loop). CUDA has not shown this failure in either an
    equivalent poison-value test or one real training run, plausibly
    because NVIDIA's decade-hardened GPUDirect RDMA stack already
    enforces this coherence transparently -- but that evidence is narrower
    than what exposed the ROCm bug (a real training loop, not just
    isolated tests), so this is treated as "no evidence of the problem on
    CUDA yet" rather than "proven unnecessary on CUDA". DDSTORE_GPU_SYNC
    exists specifically so this default can be overridden the moment
    either direction needs it, without a code change.
    """
    import torch
    override = os.environ.get("DDSTORE_GPU_SYNC", "auto").strip().lower()
    if override == "always":
        return True
    if override == "never":
        return False
    return torch.version.hip is not None

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
        void get[T](string name, long start, long count, T* buffer, int hmem_iface) except +
        void prefetch_recv_mr[T](string name, T* buffer, long nrows, int disp, int hmem_iface) except +
        void epoch_begin()
        void epoch_end()
        void free()
        void init(string name, long nrows, int disp, int itemsize) except +
        void update[T](string name, T* buffer, long nrows, long offset) except +
        void join(string name) except +
        void query(string name, VarInfo &varinfo) except +
        long size(string name) except +

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
        cdef int iface
        cdef long nrows
        cdef int disp
        if _is_cuda_tensor(arr):
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
            # before handing it to RDMA -- see _should_sync_before_rdma()
            # for what this guards against and why it's conditional.
            if _should_sync_before_rdma(arr):
                torch.cuda.synchronize(device=arr.device)
            iface = _hmem_iface_for(arr)
            ptr = arr.data_ptr()
            nrows = arr.shape[0]
            disp = arr.numel() // arr.shape[0]
            if arr.dtype == torch.int32:
                self.c_ddstore.add(s2b(name), <int *> ptr, nrows, disp, iface)
            elif arr.dtype == torch.int64:
                self.c_ddstore.add(s2b(name), <long *> ptr, nrows, disp, iface)
            elif arr.dtype == torch.uint8:
                self.c_ddstore.add(s2b(name), <char *> ptr, nrows, disp, iface)
            elif arr.dtype == torch.float32:
                self.c_ddstore.add(s2b(name), <float *> ptr, nrows, disp, iface)
            elif arr.dtype == torch.float64:
                self.c_ddstore.add(s2b(name), <double *> ptr, nrows, disp, iface)
            elif arr.dtype == torch.bool:
                self.c_ddstore.add(s2b(name), <char *> ptr, nrows, disp, iface)
            else:
                raise NotImplementedError
            # Keepalive: DDStore now holds a raw pointer into arr's storage
            # with no copy and no C++-level refcounting -- see ddstore.hpp
            # add()'s lifetime-contract doc comment. Must outlive this
            # variable's registration; cleared in free()/__dealloc__.
            self._gpu_owned_buffers[name] = arr
            return

        cdef np.ndarray np_arr = arr
        assert np_arr.flags.c_contiguous
        nrows = np_arr.shape[0]
        disp = np_arr.size // np_arr.shape[0]
        if np_arr.dtype == np.int32:
            self.c_ddstore.add(s2b(name), <int *> np_arr.data, nrows, disp, 0)
        elif np_arr.dtype == np.int64:
            self.c_ddstore.add(s2b(name), <long *> np_arr.data, nrows, disp, 0)
        elif np_arr.dtype == np.uint8:
            self.c_ddstore.add(s2b(name), <char *> np_arr.data, nrows, disp, 0)
        elif np_arr.dtype == np.float32:
            self.c_ddstore.add(s2b(name), <float *> np_arr.data, nrows, disp, 0)
        elif np_arr.dtype == np.float64:
            self.c_ddstore.add(s2b(name), <double *> np_arr.data, nrows, disp, 0)
        elif np_arr.dtype == np.bool_:
            self.c_ddstore.add(s2b(name), <char *> np_arr.data, nrows, disp, 0)
        else:
            raise NotImplementedError

    def get(self, str name, arr, long start=0):
        cdef long count = arr.shape[0]
        cdef size_t ptr
        cdef int iface
        if _is_cuda_tensor(arr):
            _check_gpu_fabric_preconditions(self.method, "GPU destination buffer")
            assert arr.is_contiguous()
            import torch
            # See _should_sync_before_rdma() / the matching comment in
            # add() for what this guards against and why it's conditional.
            if _should_sync_before_rdma(arr):
                torch.cuda.synchronize(device=arr.device)
            iface = _hmem_iface_for(arr)
            ptr = arr.data_ptr()
            if arr.dtype == torch.int32:
                self.c_ddstore.get(s2b(name), start, count, <int *> ptr, iface)
            elif arr.dtype == torch.int64:
                self.c_ddstore.get(s2b(name), start, count, <long *> ptr, iface)
            elif arr.dtype == torch.uint8:
                self.c_ddstore.get(s2b(name), start, count, <char *> ptr, iface)
            elif arr.dtype == torch.float32:
                self.c_ddstore.get(s2b(name), start, count, <float *> ptr, iface)
            elif arr.dtype == torch.float64:
                self.c_ddstore.get(s2b(name), start, count, <double *> ptr, iface)
            elif arr.dtype == torch.bool:
                self.c_ddstore.get(s2b(name), start, count, <char *> ptr, iface)
            else:
                raise NotImplementedError
            return

        cdef np.ndarray np_arr = arr
        assert np_arr.flags.c_contiguous
        assert np_arr.shape[0] >= count
        if np_arr.dtype == np.int32:
            self.c_ddstore.get(s2b(name), start, count, <int *> np_arr.data, 0)
        elif np_arr.dtype == np.int64:
            self.c_ddstore.get(s2b(name), start, count, <long *> np_arr.data, 0)
        elif np_arr.dtype == np.uint8:
            self.c_ddstore.get(s2b(name), start, count, <char *> np_arr.data, 0)
        elif np_arr.dtype == np.float32:
            self.c_ddstore.get(s2b(name), start, count, <float *> np_arr.data, 0)
        elif np_arr.dtype == np.float64:
            self.c_ddstore.get(s2b(name), start, count, <double *> np_arr.data, 0)
        elif np_arr.dtype == np.bool_:
            self.c_ddstore.get(s2b(name), start, count, <char *> np_arr.data, 0)
        else:
            raise NotImplementedError
    
    def epoch_begin(self):
        self.c_ddstore.epoch_begin()

    def epoch_end(self):
        self.c_ddstore.epoch_end()

    def free(self):
        self.c_ddstore.free()
        self._gpu_owned_buffers.clear()

    def prefetch_recv_mr(self, str name, arr):
        """Pre-register a GPU tensor as the recv MR for variable `name`.

        Call once with the full pool tensor (e.g. shape [POOL, disp]) before
        the first get() call.  read_from_remote() will reuse this registration
        for any buffer pointer that falls within the registered region, so all
        pool slices share one fi_mr_regattr call instead of one per slice.
        No-op for host (non-CUDA) tensors.
        """
        if not _is_cuda_tensor(arr):
            return  # host path: no pre-registration needed
        _check_gpu_fabric_preconditions(self.method, "GPU recv pool")
        assert arr.is_contiguous()
        import torch
        cdef size_t ptr = arr.data_ptr()
        cdef long nrows = arr.shape[0]
        cdef int disp   = arr.numel() // arr.shape[0]
        cdef int iface  = _hmem_iface_for(arr)
        if arr.dtype == torch.float32:
            self.c_ddstore.prefetch_recv_mr(s2b(name), <float *> ptr, nrows, disp, iface)
        elif arr.dtype == torch.float64:
            self.c_ddstore.prefetch_recv_mr(s2b(name), <double *> ptr, nrows, disp, iface)
        elif arr.dtype == torch.int32:
            self.c_ddstore.prefetch_recv_mr(s2b(name), <int *> ptr, nrows, disp, iface)
        elif arr.dtype == torch.int64:
            self.c_ddstore.prefetch_recv_mr(s2b(name), <long *> ptr, nrows, disp, iface)
        else:
            raise NotImplementedError("prefetch_recv_mr: unsupported dtype %s" % arr.dtype)

    def init(self, str name, long nrows, int disp, int itemsize=1):
        self.c_ddstore.init(s2b(name), nrows, disp, itemsize)

    def update(self, str name, arr, long offset):
        if _is_cuda_tensor(arr):
            raise NotImplementedError(
                "GPU source buffers are not yet supported by update() "
                "(GPU-to-GPU is a future phase); pass arr.cpu().numpy() instead")
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
