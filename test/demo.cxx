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
    int *len;
    int N = 4;
    buffer = (double *)malloc(N * sizeof(double));
    len = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        buffer[i] = i + 1 + 10 * rank;
        len[i] = 1;
        printf("%d: buffer[%d] = %g\n", rank, i, buffer[i]);
    }

    DDStore dds(comm);
    dds.create("var", buffer, 1, len, N);

    double getbuf[4] = {0.0, 0.0, 0.0, 0.0};
    int id = (rank * N + N) % (N * comm_size);
    printf("[%d] id: %d\n", rank, id);
    dds.get("var", id, getbuf);

    printf("%d: %g %g %g %g\n", rank, getbuf[0], getbuf[1], getbuf[2], getbuf[3]);

    MPI_Finalize();
    return EXIT_SUCCESS;
}