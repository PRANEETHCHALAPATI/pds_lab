#include <stdio.h>
#include <math.h>
#define N 16
#define TPB 8
#define MIN_PART 16

// Insertion sort for small partitions
__device__ void insertionSort(int *a, int l, int r) {
    for (int i = l + 1; i <= r; i++) {
        int key = a[i], j = i - 1;
        while (j >= l && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// Partition the array
__device__ int partition(int *a, int l, int r) {
    int p = a[r], i = l - 1;
    for (int j = l; j < r; j++)
        if (a[j] < p) {
            i++;
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
    int t = a[i + 1]; a[i + 1] = a[r]; a[r] = t;
    return i + 1;
}

// Kernel: parallel partitioning
__global__ void quicksortKernel(int *a, int *l, int *r, int *nl, int *nr, int nTasks, int *next) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nTasks) return;

    int low = l[id], high = r[id];
    if (low < high) {
        if (high - low + 1 <= MIN_PART) {
            insertionSort(a, low, high);
            return;
        }

        int p = partition(a, low, high);
        if (p - 1 > low) {
            int idx = atomicAdd(next, 1);
            nl[idx] = low;
            nr[idx] = p - 1;
        }
        if (p + 1 < high) {
            int idx = atomicAdd(next, 1);
            nl[idx] = p + 1;
            nr[idx] = high;
        }
    }
}

int main() {
    int h_a[N] = {24,17,85,13,9,54,76,45,4,63,21,33,89,12,99,1};
    int *d_a; cudaMalloc(&d_a, N*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Device buffers for tasks
    int *l, *r, *nl, *nr, *next;
    cudaMalloc(&l, N*sizeof(int)); cudaMalloc(&r, N*sizeof(int));
    cudaMalloc(&nl, N*sizeof(int)); cudaMalloc(&nr, N*sizeof(int));
    cudaMalloc(&next, sizeof(int));

    // Initialize first task
    int h_l[1] = {0}, h_r[1] = {N-1};
    cudaMemcpy(l, h_l, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(r, h_r, sizeof(int), cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Iteratively launch kernels
    int nTasks = 1;
    while (nTasks > 0) {
        cudaMemset(next, 0, sizeof(int));
        int blocks = (nTasks + TPB - 1) / TPB;
        quicksortKernel<<<blocks, TPB>>>(d_a, l, r, nl, nr, nTasks, next);
        cudaDeviceSynchronize();
        cudaMemcpy(&nTasks, next, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap buffers for next iteration
        int *tmpL = l; l = nl; nl = tmpL;
        int *tmpR = r; r = nr; nr = tmpR;
    }

    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    // Copy back sorted array
    cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate numerical complexities
    double log2n = log2((double)N);
    double timec = ((double)N * log2n) / TPB;
    double costc = (double)N * log2n;

    // Output sorted array
    printf("Sorted Array: ");
    for (int i = 0; i < N; i++) printf("%d ", h_a[i]);
    printf("\n\n===== COMPLEXITY ANALYSIS =====\n");
    printf("Time Complexity : O((N log N)/P) = %.2f units\n", timec);
    printf("Cost Complexity : O(N log N) = %.2f units\n", costc);
    printf("Execution Time : %.5f ms\n", ms);

    // Cleanup
    cudaFree(d_a); cudaFree(l); cudaFree(r);
    cudaFree(nl); cudaFree(nr); cudaFree(next);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
