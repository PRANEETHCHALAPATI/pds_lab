#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define MAX_MSG_LEN 100

int main(int argc, char *argv[]) {
    int rank, size;
    char message[MAX_MSG_LEN];
    MPI_Status status;
    int i;
   
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
   
    // Root process prepares the broadcast message
    if (rank == 0) {
        strcpy(message, "Hello from root!");
        printf("Process %d broadcasting message: %s\n", rank, message);
    }
   
    // All processes participate in the broadcast
    MPI_Bcast(message, MAX_MSG_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
   
    // Non-root processes display the received broadcast message
    if (rank != 0) {
        printf("Process %d received broadcast message: %s\n", rank, message);
    }
   
    // Non-root processes send messages to root
    if (rank != 0) {
        sprintf(message, "Hello from process %d", rank);
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    else {
        // Root process receives messages from all other processes
        for (i = 1; i < size; i++) {
            MPI_Recv(message, MAX_MSG_LEN, MPI_CHAR, i, 1, MPI_COMM_WORLD, &status);
            printf("Root process received message: %s\n", message);
        }
    }
   
    // Additional communication between non-root processes
    if (rank == 1 && size > 2) {
        // Process 1 sends message to process 2
        sprintf(message, "Message from process 1 to process 2");
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 2, 2, MPI_COMM_WORLD);
    }
    else if (rank == 2) {
        // Process 2 receives message from process 1
        MPI_Recv(message, MAX_MSG_LEN, MPI_CHAR, 1, 2, MPI_COMM_WORLD, &status);
        printf("Process %d received message: %s\n", rank, message);
    }
   
    // Finalize MPI
    MPI_Finalize();
    return 0;
}