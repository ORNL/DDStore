import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import pytest
from mpi4py import MPI


@pytest.fixture(scope="function")
def comm():
    return MPI.COMM_WORLD
