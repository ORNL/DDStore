# distutils: language=c++
# cython: language_level=3
# cython: language=c++

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
        void add[T](string name, T* buffer, long nrows, int disp) except +
        void get[T](string name, long start, long count, T* buffer) except +
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
            mpi_comm = comm_or_none
            self.c_ddstore = new DDStore(method, mpi_comm.ob_mpi)

    def __dealloc__(self):
        if self.c_ddstore != NULL:
            del self.c_ddstore
            self.c_ddstore = NULL

    def add(self, str name, np.ndarray arr):
        assert arr.flags.c_contiguous
        cdef long nrows = arr.shape[0]
        cdef int disp = arr.size // arr.shape[0]
        if arr.dtype == np.int32:
            self.c_ddstore.add(s2b(name), <int *> arr.data, nrows, disp)
        elif arr.dtype == np.int64:
            self.c_ddstore.add(s2b(name), <long *> arr.data, nrows, disp)
        elif arr.dtype == np.uint8:
            self.c_ddstore.add(s2b(name), <char *> arr.data, nrows, disp)
        elif arr.dtype == np.float32:
            self.c_ddstore.add(s2b(name), <float *> arr.data, nrows, disp)
        elif arr.dtype == np.float64:
            self.c_ddstore.add(s2b(name), <double *> arr.data, nrows, disp)
        elif arr.dtype == np.bool_:
            self.c_ddstore.add(s2b(name), <char *> arr.data, nrows, disp)
        else:
            raise NotImplementedError

    def get(self, str name, np.ndarray arr, long start=0):
        assert arr.flags.c_contiguous
        cdef long count = arr.shape[0]
        assert arr.shape[0] >= count
        if arr.dtype == np.int32:
            self.c_ddstore.get(s2b(name), start, count, <int *> arr.data)
        elif arr.dtype == np.int64:
            self.c_ddstore.get(s2b(name), start, count, <long *> arr.data)
        elif arr.dtype == np.uint8:
            self.c_ddstore.get(s2b(name), start, count, <char *> arr.data)
        elif arr.dtype == np.float32:
            self.c_ddstore.get(s2b(name), start, count, <float *> arr.data)
        elif arr.dtype == np.float64:
            self.c_ddstore.get(s2b(name), start, count, <double *> arr.data)
        elif arr.dtype == np.bool_:
            self.c_ddstore.get(s2b(name), start, count, <char *> arr.data)
        else:
            raise NotImplementedError
    
    def epoch_begin(self):
        self.c_ddstore.epoch_begin()

    def epoch_end(self):
        self.c_ddstore.epoch_end()

    def free(self):
        self.c_ddstore.free()
    
    def init(self, str name, long nrows, int disp, int itemsize=1):
        self.c_ddstore.init(s2b(name), nrows, disp, itemsize)

    def update(self, str name, np.ndarray arr, long offset):
        assert arr.flags.c_contiguous
        cdef long nrows = arr.shape[0]
        if arr.dtype == np.int32:
            self.c_ddstore.update(s2b(name), <int *> arr.data, nrows, offset)
        elif arr.dtype == np.int64:
            self.c_ddstore.update(s2b(name), <long *> arr.data, nrows, offset)
        elif arr.dtype == np.uint8:
            self.c_ddstore.update(s2b(name), <char *> arr.data, nrows, offset)
        elif arr.dtype == np.float32:
            self.c_ddstore.update(s2b(name), <float *> arr.data, nrows, offset)
        elif arr.dtype == np.float64:
            self.c_ddstore.update(s2b(name), <double *> arr.data, nrows, offset)
        elif arr.dtype == np.bool_:
            self.c_ddstore.update(s2b(name), <char *> arr.data, nrows, offset)
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
