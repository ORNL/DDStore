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
# from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE

import io
import pickle

from cpython cimport array
import array
from libc.stdlib cimport malloc, free

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
        DDStore(libmpi.MPI_Comm comm, int use_mq, int role)
        void create[T](string name, T* buffer, int disp, int* local_lenlist, int ncount) except +
        int get[T](string name, long id, T* buffer, int size) except +
        void epoch_begin()
        void epoch_end()
        void free()
        void query(string name, VarInfo &varinfo)
        int use_mq
        int role

cdef class PyDDstoreVarinfo:
    cdef VarInfo c_varinfo

    def __cinit__(self):
        pass

cdef class PyDDStore:
    cdef DDStore c_ddstore
    cdef dict buffer_list

    def __cinit__(self, MPI.Comm comm, use_mq=0, role=0):
        self.c_ddstore = DDStore(comm.ob_mpi, use_mq, role)
        self.buffer_list = dict()

    cpdef test(self):
        cdef array.array a = array.array('i', [])
        array.resize(a, 10)
        print (a, len(a))

    cpdef add(self, str name, input, lenlist=None):
        cdef int[:] lenarr
        if isinstance(input, io.BytesIO):
            buffer = input.getbuffer()
            lenarr = array.array('i', lenlist)
            self.__create_from_buffer(name, buffer, lenarr)
        elif isinstance(input, memoryview):
            lenarr = array.array('i', lenlist)
            self.__create_from_buffer(name, input, lenarr)
        elif isinstance(input, np.ndarray):
            self.__create__from_ndarray(name, input, lenarr)
        elif isinstance(input, list):
            buffer = io.BytesIO()
            self.buffer_list[name] = buffer

            lenlist = list()
            prev = 0
            for i in range(len(input)):
                pickle.dump(input[i], buffer)
                lenlist.append(buffer.getbuffer().nbytes - prev)
                prev = buffer.getbuffer().nbytes
            lenarr = array.array('i', lenlist)
            self.__create_from_buffer(name, buffer.getbuffer(), lenarr)
        else:
            raise NotImplementedError

    cpdef __create_from_buffer(self, str name, char [:] buffer, int [:] lenlist):
        cdef char *ptr = &buffer[0]
        cdef int *plen = &lenlist[0]
        cdef int ncount = len(lenlist)
        cdef int dist = 1
        self.c_ddstore.create(s2b(name), <char *> ptr, dist, <int *> plen, ncount)

    def __create__from_ndarray(self, str name, np.ndarray arr, np.ndarray lenlist):
        assert arr.flags.c_contiguous
        assert lenlist.flags.c_contiguous
        assert lenlist.dtype == np.int32
        assert lenlist.ndim == 1
        assert lenlist.sum() == arr.shape[0]
        cdef long nrows = arr.shape[0]
        cdef int disp = arr.size // arr.shape[0]
        cdef int ncount = lenlist.size
        if arr.dtype == np.int32:
            self.c_ddstore.create(s2b(name), <int *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.int64:
            self.c_ddstore.create(s2b(name), <long *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.float32:
            self.c_ddstore.create(s2b(name), <float *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.float64:
            self.c_ddstore.create(s2b(name), <double *> arr.data, disp, <int *> lenlist.data, ncount)
        elif arr.dtype == np.dtype('S1'):
            self.c_ddstore.create(s2b(name), <char *> arr.data, disp, <int *> lenlist.data, ncount)
        else:
            raise NotImplementedError

    def get(self, str name, long id, decoder=None):
        cdef np.ndarray arr
        if (not self.c_ddstore.use_mq) or (self.c_ddstore.use_mq and self.c_ddstore.role):
            if id >= 0:
                n = self.query(name, id)
                print ("query:", id, n)
                arr = np.chararray(n)
                self.c_ddstore.get(s2b(name), id, <char *> arr.data, arr.size)
                rtn = arr.data[:n]
                if decoder is not None:
                    rtn = decoder(rtn)
            else:
                rtn = self.c_ddstore.get(s2b(name), -1, <char *> NULL, 0)
        else:
            rtn = self.c_ddstore.get(s2b(name), -1, <char *> NULL, 0)
        return rtn

    def get_ndarray(self, str name, np.ndarray arr, long id):
        assert arr.flags.c_contiguous
        cdef long count = arr.shape[0]
        assert arr.shape[0] >= count
        if arr.dtype == np.int32:
            self.c_ddstore.get(s2b(name), id, <int *> arr.data, arr.size)
        elif arr.dtype == np.int64:
            self.c_ddstore.get(s2b(name), id, <long *> arr.data, arr.size)
        elif arr.dtype == np.float32:
            self.c_ddstore.get(s2b(name), id, <float *> arr.data, arr.size)
        elif arr.dtype == np.float64:
            self.c_ddstore.get(s2b(name), id, <double *> arr.data, arr.size)
        elif arr.dtype == np.dtype('S1'):
            self.c_ddstore.get(s2b(name), id, <char *> arr.data, arr.size)
        else:
            raise NotImplementedError
    
    def query(self, str name, long id):
        cdef VarInfo varinfo
        self.c_ddstore.query(s2b(name), varinfo)
        return varinfo.lenlist[id]
    
    def buffer(self, str name):
        return self.buffer_list[name]

    def epoch_begin(self):
        self.c_ddstore.epoch_begin()

    def epoch_end(self):
        self.c_ddstore.epoch_end()

    def free(self):
        self.c_ddstore.free()
