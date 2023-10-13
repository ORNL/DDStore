#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "ddstore.hpp"

int main(int argc, char *argv[])
{
    if (argc > 4)
    {
        fprintf(stderr, "Usage: %s <nsample> <use_mq> <role>\n", argv[0]);
        fprintf(stderr, "  nsample: number of samples per process\n");
        fprintf(stderr, "  use_mq: 0 = false, 1 = true\n");
        fprintf(stderr, "  role: 0 = producer, 1 = consumer\n");
        return EINVAL;
    }

    int N = 10;
    int use_mq = 0;
    int role = 0;

    if (argc > 1)
        N = atoi(argv[1]);

    if (argc > 2)
        use_mq = atoi(argv[2]);

    if (argc > 3)
        role = atoi(argv[3]);
    printf("nsample,use_mq,role: %d %d %d\n", N, use_mq, role);

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm, &rank);

    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    double *buffer;
    int *len;
    buffer = (double *)malloc(N * sizeof(double));
    len = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        buffer[i] = i + N * rank + 1;
        if (role == 1) 
            buffer[i] *= -1;
        len[i] = 1;
        printf("[%d:%d] buffer[%d] = %g\n", role, rank, i, buffer[i]);
    }

    DDStore dds(comm, use_mq, role);
    dds.create<double>("var", buffer, 1, len, N);

    int ntotal = N * comm_size;
    double getbuf[ntotal];
    for (int i = 0; i < ntotal; i++)
    {
        printf("[%d:%d] reading: %d\n", role, rank, ntotal-i-1);
        dds.get<double>("var", ntotal-i-1, &(getbuf[i]), 1);
    }
    for (int i = 0; i < ntotal; i++)
    {
        printf("[%d:%d] received[%d]: %g\n", role, rank, i, getbuf[i]);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
