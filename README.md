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

* MQ producer/consumer test
```
mpirun -n 2 python test/demo2.py --mq --consumer & 
mpirun -n 2 python test/demo2.py --mq --producer
```
