#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include<mpi.h>

#define REQUEST 1
#define REPLY 2
#define RELEASE 3
#define MAX_PROCESSES 10

int timestamp = 0;
int reply_count = 0;
bool requesting_cs = false;
bool in_cs = false;

int num_processes, rank;
int deferred[MAX_PROCESSES] = {0};

// Increment Lamport clock
int increment_clock(int received_time) {
    if (received_time > timestamp)
        timestamp = received_time;
    return ++timestamp;
}

// Broadcast a REQUEST to all other processes
void send_request() {
    timestamp++;
    requesting_cs = true;
    reply_count = 0;
    for (int i = 0; i < num_processes; i++) {
        if (i != rank) {
            MPI_Send(&timestamp, 1, MPI_INT, i, REQUEST, MPI_COMM_WORLD);
            printf("Process %d -> Sent REQUEST to %d (timestamp %d)\n", rank, i, timestamp);
        }
    }
}

// Enter Critical Section
void enter_cs() {
    in_cs = true;
    printf(">>> Process %d ENTERED critical section (timestamp %d)\n", rank, timestamp);
    sleep(1); // simulate work
    printf("<<< Process %d EXITING critical section\n", rank);
    in_cs = false;
    requesting_cs = false;

    // Send deferred replies
    for (int i = 0; i < num_processes; i++) {
        if (deferred[i]) {
            deferred[i] = 0;
            MPI_Send(&timestamp, 1, MPI_INT, i, REPLY, MPI_COMM_WORLD);
            printf("Process %d -> Sent deferred REPLY to %d\n", rank, i);
        }
    }
}

// Receive and handle messages
void handle_messages() {
    MPI_Status status;
    int msg_time;

    while (reply_count < num_processes - 1) {
        MPI_Recv(&msg_time, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        timestamp = increment_clock(msg_time);

        switch (status.MPI_TAG) {
            case REQUEST:
                if (in_cs || (requesting_cs && (msg_time > timestamp))) {
                    deferred[status.MPI_SOURCE] = 1;
                    printf("Process %d DEFERRED reply to %d\n", rank, status.MPI_SOURCE);
                } else {
                    MPI_Send(&timestamp, 1, MPI_INT, status.MPI_SOURCE, REPLY, MPI_COMM_WORLD);
                    printf("Process %d -> Sent REPLY to %d\n", rank, status.MPI_SOURCE);
                }
                break;

            case REPLY:
                reply_count++;
                printf("Process %d received REPLY from %d (%d/%d)\n", rank, status.MPI_SOURCE, reply_count, num_processes - 1);
                break;
        }

        if (reply_count == num_processes - 1 && requesting_cs)
            enter_cs();
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(rank + time(NULL));
    sleep(rank); // Stagger requests for clarity

    if (rank == 0)
        printf("\n=== Ricart-Agrawala Distributed Mutual Exclusion ===\n\n");

    send_request();
    handle_messages();

    MPI_Finalize();
    return 0;
}