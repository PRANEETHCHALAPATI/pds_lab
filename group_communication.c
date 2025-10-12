#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int bcast_data;
    int scatter_data[4] = {10, 20, 30, 40};
    int recv_scatter;
    int gather_data[4];
    int reduce_result;

   
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   
    if (rank == 0) {
        bcast_data = 100;
        printf("Process %d broadcasts %d\n", rank, bcast_data);
    }
    MPI_Bcast(&bcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        printf("Process %d received %d from bcast\n", rank, bcast_data);
    }

   
    MPI_Scatter(scatter_data, 1, MPI_INT, &recv_scatter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received %d from scatter\n", rank, recv_scatter);

   
    MPI_Gather(&recv_scatter, 1, MPI_INT, gather_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Root process gathered data: ");
        for (int i = 0; i < size; i++) {
            printf("%d ", gather_data[i]);
        }
        printf("\n");
    }

 
    MPI_Reduce(&recv_scatter, &reduce_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reduction result: %d\n", reduce_result);
    }

   
    MPI_Finalize();
    return 0;
}