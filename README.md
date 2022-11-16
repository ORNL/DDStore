# Distributed Data Store

This is a C++ library to implement a distributed dataset. It provides a python binding.

## Installation
```
CC=mpicc CXX=mpicxx python setup.py build_ext --inplace
```

## Test
```
mpirun -n 4 python test/demo.py
```
