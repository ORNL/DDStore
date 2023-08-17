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
from libcpp.vector cimport vector

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
        string typeinfo
        int disp
        vector[int] lenlist


    cdef cppclass DDStore:
        DDStore()
        DDStore(libmpi.MPI_Comm comm)
        void create[T](string name, T* buffer, int disp, int* local_lenlist, int ncount) except +
        void get[T](string name, long id, T* buffer) except +
        void epoch_begin()
        void epoch_end()
        void free()

cdef class PyDDstoreVarinfo:
    cdef VarInfo c_varinfo

    def __cinit__(self):
        pass

cdef class PyDDStore:
    cdef DDStore c_ddstore

    def __cinit__(self, MPI.Comm comm):
        self.c_ddstore = DDStore(comm.ob_mpi)
    
    def create(self, str name, np.ndarray arr, np.ndarray lenlist):
        assert arr.flags.c_contiguous
        assert lenlist.flags.c_contiguous
        assert lenlist.dtype == np.int32
        assert lenlist.ndim == 1
        assert lenlist.sum() == arr.shape[0]
        cdef long nrows = arr.shape[0]
        cdef int disp = arr.size // arr.shape[0]
        cdef ncount = lenlist.size
        if arr.dtype == np.int32:
            self.c_ddstore.create(s2b(name), <int *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.int64:
            self.c_ddstore.create(s2b(name), <long *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.float32:
            self.c_ddstore.create(s2b(name), <float *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.float64:
            self.c_ddstore.create(s2b(name), <double *> arr.data, disp, <int *> lenlist.data, ncount)
        else:
            raise NotImplementedError

    def get(self, str name, np.ndarray arr, long id):
        assert arr.flags.c_contiguous
        cdef long count = arr.shape[0]
        assert arr.shape[0] >= count
        if arr.dtype == np.int32:
            self.c_ddstore.get(s2b(name), id, <int *> arr.data)
        elif arr.dtype == np.int64:
            self.c_ddstore.get(s2b(name), id, <long *> arr.data)
        elif arr.dtype == np.float32:
            self.c_ddstore.get(s2b(name), id, <float *> arr.data)
        elif arr.dtype == np.float64:
            self.c_ddstore.get(s2b(name), id, <double *> arr.data)
        else:
            raise NotImplementedError
    
    def epoch_begin(self):
        self.c_ddstore.epoch_begin()

    def epoch_end(self):
        self.c_ddstore.epoch_end()

    def free(self):
        self.c_ddstore.free()
