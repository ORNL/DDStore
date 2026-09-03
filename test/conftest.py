import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import pytest
from mpi4py import MPI


@pytest.fixture(scope="function")
def comm():
    """Provide MPI.COMM_WORLD and barrier after each test.

    The barrier ensures all ranks finish the current test (including
    store.free() and any fabric endpoint teardown) before any rank
    begins the next test's handshake/MPI_Allgather.  Without this,
    CXI endpoint cleanup on a fast rank can desynchronize the ranks
    enough to deadlock the next test's collective in add().
    """
    yield MPI.COMM_WORLD
    MPI.COMM_WORLD.Barrier()
