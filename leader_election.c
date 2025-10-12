#include <stdio.h>
#include <mpi.h>

#define TAG_ELECTION 1
#define TAG_LEADER 2

int main(int argc, char *argv[]) {
    int rank, size;
    int leader, recv_rank;
    MPI_Status status;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Need at least 2 processes.\n");
        MPI_Finalize();
        return 0;
    }

    int initiator = 0;
    int highest_rank = rank;

    if (rank == initiator) {
        printf("Process %d detected leader failure and initiated election.\n", rank);
        MPI_Send(&rank, 1, MPI_INT, (rank + 1) % size, TAG_ELECTION, MPI_COMM_WORLD);
        MPI_Recv(&recv_rank, 1, MPI_INT, (rank - 1 + size) % size, TAG_ELECTION, MPI_COMM_WORLD, &status);
       
        if (recv_rank > highest_rank)
            highest_rank = recv_rank;

        leader = highest_rank;
        printf("Process %d announces new leader: %d\n", rank, leader);

        for (i = 1; i < size; i++)
            MPI_Send(&leader, 1, MPI_INT, i, TAG_LEADER, MPI_COMM_WORLD);
    }
    else {
        MPI_Recv(&recv_rank, 1, MPI_INT, (rank - 1 + size) % size, TAG_ELECTION, MPI_COMM_WORLD, &status);

        if (recv_rank > highest_rank)
            highest_rank = recv_rank;
        if (rank != initiator)
            MPI_Send(&highest_rank, 1, MPI_INT, (rank + 1) % size, TAG_ELECTION, MPI_COMM_WORLD);

        MPI_Recv(&leader, 1, MPI_INT, initiator, TAG_LEADER, MPI_COMM_WORLD, &status);
        printf("Process %d received leader: %d\n", rank, leader);
    }

    MPI_Finalize();
    return 0;
}