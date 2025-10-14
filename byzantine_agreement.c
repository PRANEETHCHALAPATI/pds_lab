#include <stdio.h>
#include <mpi.h>
#include <time.h>

#define N 4

int main(int argc, char *argv[]) {
int i;
int sender;
int rank, size;
int value;
int received[N];
int received_matrix[N][N]; // stores what each process received from each other
int faulty_rank = 3; // process 3 acts Byzantine

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

if (size != N) {
if (rank == 0)
printf("Run with %d processes only!\n", N);
MPI_Finalize();
return 0;
}

srand(time(NULL) + rank);

// Step 1: Assign random initial value (0 or 1)
value = rand() % 2;
printf("Process %d initial value: %d\n", rank, value);

MPI_Barrier(MPI_COMM_WORLD);

// Step 2: Send values to all other processes (no self-loop)
for (i = 0; i < size; i++) {
if (i == rank) continue; // skip self
int send_value = value;

// Byzantine process sends inconsistent values
if (rank == faulty_rank) {
send_value = rand() % 2;
printf("Process %d (faulty) sends %d to Process %d\n", rank, send_value, i);
} else {
printf("Process %d sends %d to Process %d\n", rank, send_value, i);
}

MPI_Send(&send_value, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
}

// Step 3: Receive values from all others (no self-loop)
for (i = 0; i < size; i++) {
if (i == rank) {
received[i] = value;
continue;
}
MPI_Recv(&received[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD,
MPI_STATUS_IGNORE);
}

// Print what each process received
printf("\nProcess %d received values: ", rank);

for (i = 0; i < size; i++)
printf("%d ", received[i]);
printf("\n");

// Step 4: Gather all received data at every process
MPI_Allgather(received, N, MPI_INT, received_matrix, N, MPI_INT,
MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);

// Step 5: Identify inconsistent sender (faulty node)
if (rank == 0) {
printf("\n--- Fault Detection Report ---\n");
for (sender = 0; sender < N; sender++) {
int consistent = 1;
for (i = 1; i < N; i++) {
if (received_matrix[i][sender] != received_matrix[0][sender])
consistent = 0;
}

if (!consistent)
printf(" Process %d is detected as FAULTY (sent inconsistent values)\n", sender);
else
printf(" Process %d is consistent across all receivers\n", sender);
}
}

MPI_Finalize();
return 0;
}