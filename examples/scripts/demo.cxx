#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "ddstore.hpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    double *buffer;
    const int N = 4;
    buffer = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        buffer[i] = i + 1 + 10 * rank;
        printf("%d: buffer[%d] = %g\n", rank, i, buffer[i]);
    }

    int method = 1; // 0: MPI, 1: libfabric
    DDStore ds(method, comm);
    ds.add("var", buffer, 2, 2);

    double getbuf[4] = {0.0, 0.0, 0.0, 0.0};
    int start = (2 * (rank + 1)) % (2 * comm_size) + 1;
    printf("start: %d\n", start);
    ds.get("var", start, 1, getbuf);

    printf("%d: %g %g %g %g\n", rank, getbuf[0], getbuf[1], getbuf[2], getbuf[3]);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
