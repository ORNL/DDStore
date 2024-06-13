# DDStore

Enabling efficient distributed data loading for distributed data parallelism

<img src="https://github.com/allaffa/DDStore/assets/2488656/88a3b139-062d-41e8-a8d7-40c1a144d897" alt="HydraGNN_QRcode" width="300" />


## Installation
```
pip install numpy mpi4py Cython # install pre-requisites

# for use with PYTHONPATH=$PWD:$PYTHONPATH
CC=mpicc CXX=mpicxx python setup.py build_ext --inplace

# or, for installing into your virtual environment using pip
CC=mpicc CXX=mpicxx pip install .

# or, as above but in development mode
CC=mpicc CXX=mpicxx pip install -e .
```

## Test
```
mpirun -n 4 python -m mpi4py test/demo.py
```
