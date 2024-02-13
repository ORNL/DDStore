# DDStore
Enabling efficient distributed data loading for distributed data parallelism

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="HydraGNN_QRcode" width="300" />


## Installation
```
CC=mpicc CXX=mpicxx python setup.py build_ext --inplace
```

## Test
```
mpirun -n 4 python test/demo.py
```
