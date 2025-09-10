#include <stdio.h>
#include <cuda_runtime.h>

#define N 20 // Number of elements to sort

// CUDA kernel for odd-even transposition sort
__global__ void oddEvenTS(int *data, int n, int phase) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n - 1) {
        // Odd phase: compare (1,2), (3,4), ...
        if (phase % 2 == 0 && (tid % 2 == 0)) {
            if (data[tid] > data[tid + 1]) {
                int temp = data[tid];
                data[tid] = data[tid + 1];
                data[tid + 1] = temp;
            }
        }
        // Even phase: compare (0,1), (2,3), ...
        if (phase % 2 == 1 && (tid % 2 == 1)) {
            if (data[tid] > data[tid + 1]) {
                int temp = data[tid];
                data[tid] = data[tid + 1];
                data[tid + 1] = temp;
            }
        }
    }
}

int main() {
    int h_data[N] = {9, 4, 8, 3, 1, 2, 7, 6, 5, 0, 
                     12, 57, 89, 65, 42, 36, 71, 99, 87, 20};

    int *d_data;

    printf("Original array: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_data[i]);
    printf("\n");

    // Allocate memory on device
    cudaMalloc((void **)&d_data, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA event objects for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configure CUDA kernel launch
    int threadsPerBlock = 10;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start time
    cudaEventRecord(start);

    // Perform N phases of sorting
    for (int phase = 0; phase < N; phase++) {
        oddEvenTS<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, phase);
        cudaDeviceSynchronize();
    }

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy sorted data back to host
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted array
    printf("Sorted array: ");
    for (int i = 0; i < N; i++)
        printf("%d ", h_data[i]);
    printf("\n");

    // Print theoretical complexities
    printf("\n===== COMPLEXITY ANALYSIS =====\n");
    printf("Time Complexity (Parallel) : O(N) phases ≈ %d phases\n", N);
    printf("Cost Complexity : O(N * N) = O(N^2) ≈ %d operations\n", N * N);
    printf("Actual Execution Time : %.5f ms\n", elapsedTime);

    // Free device memory
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
