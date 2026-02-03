# distutils: language=c++
# cython: language_level=3
# cython: language=c++

import mpi4py.MPI as MPI
cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi

import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp.typeinfo cimport type_info

from cpython.version cimport PY_MAJOR_VERSION

cdef extern from "string.h" nogil:
    char   *strdup  (const char *s)
    size_t strlen   (const char *s)

cpdef str b2s(bytes x):
    if PY_MAJOR_VERSION < 3:
        return str(x)
    else:
        return x.decode()

cpdef bytes s2b(str x):
    if PY_MAJOR_VERSION < 3:
        return <bytes>x
    else:
        return strdup(x.encode())

cdef extern from "ddstore.hpp":
    ctypedef struct VarInfo:
        string name
        int disp
        int itemsize


    cdef cppclass DDStore:
        DDStore()
        DDStore(libmpi.MPI_Comm comm)
        DDStore(int method, libmpi.MPI_Comm comm)
        void add[T](string name, T* buffer, long nrows, int disp) except +
        void get[T](string name, long start, long count, T* buffer) except +
        void epoch_begin()
        void epoch_end()
        void free()
        void init(string name, long nrows, int disp, int itemsize) except +
        void update[T](string name, T* buffer, long nrows, long offset) except +

cdef class PyDDstoreVarinfo:
    cdef VarInfo c_varinfo

    def __cinit__(self):
        pass

cdef class PyDDStore:
    cdef DDStore c_ddstore

    def __cinit__(self, MPI.Comm comm, int method = 0):
        print("PyDDStore init method:", method)
        self.c_ddstore = DDStore(method, comm.ob_mpi)
    
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
        elif arr.dtype == np.bool:
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
        elif arr.dtype == np.bool:
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
        elif arr.dtype == np.bool:
            self.c_ddstore.update(s2b(name), <char *> arr.data, nrows, offset)
        else:
            raise NotImplementedError
