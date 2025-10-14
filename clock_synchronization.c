#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc,char** argv){
    int i;
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    
    srand((unsigned)(time(NULL) + rank * 31));
    int offset = (rand() % 21) - 10;
    time_t real_time = time(NULL);
    time_t local_time = real_time + offset;

    if(rank==0) {
        printf("Master collecting times (simulated offsets):\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
    char buf[256];
    snprintf(buf, sizeof(buf), "Process %d local time (before): %ld (offset %d)\n",
             rank, (long)local_time, offset);
    fputs(buf, stdout);
    fflush(stdout);

    if(rank==0){
    
        long *times = malloc(sizeof(long)*size);
        times[0] = (long)local_time;
        for(i=1;i<size;i++){
            MPI_Status st;
            long t;
            MPI_Recv(&t,1,MPI_LONG,i,100+i,MPI_COMM_WORLD,&st);
            times[i] = t;
        }

        
        long sum = 0;
        for(i=0;i<size;i++) sum += times[i];
        long avg = sum / size;
        printf("\nMaster computed average time: %ld\n", (long)avg);

       
        for(i=0;i<size;i++){
            long adjust = avg - times[i];
            if(i==0) {
            
                long newt = times[0] + adjust;
                printf("Master adjustment for self: %+ld -> new time %ld\n", adjust, newt);
            } else {
                MPI_Send(&adjust,1,MPI_LONG,i,200+i,MPI_COMM_WORLD);
            }
        }

       
        free(times);
    } else {
    
        long lt = (long)local_time;
        MPI_Send(&lt,1,MPI_LONG,0,100+rank,MPI_COMM_WORLD);


        long adjust;
        MPI_Status st;
        MPI_Recv(&adjust,1,MPI_LONG,0,200+rank,MPI_COMM_WORLD,&st);
        long new_local = (long)local_time + adjust;
        printf("Process %d adjustment: %+ld -> new time %ld\n", rank, adjust, new_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
