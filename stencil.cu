#include <stdio.h>
#include <cuda_runtime.h>
#include<sys/time.h>

#define N 8
#define RADIUS 1

double cpusecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Constant memory for stencil weights
__constant__ int d_weights[2 * RADIUS + 1]; // window size = 2*RADIUS + 1 = 3

// Kernel using constant memory
__global__ void stencil1D(int *in, int *out) {
    int i = threadIdx.x;  // Each thread processes one element

    int result = 0;
    for (int j = -RADIUS; j <= RADIUS; j++) {
        int idx = i + j;
        // Handle boundary conditions
        if (idx >= 0 && idx < N) {
            result += d_weights[j + RADIUS] * in[idx];
        }
    }

    out[i] = result;
}

int main() {
    int h_in[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_out[N];

    int *d_in, *d_out;

    // Define stencil weights (e.g., [1, 1, 1])
    int h_weights[2 * RADIUS + 1] = {1, 1, 1};

    // Copy weights to constant memory
    cudaMemcpyToSymbol(d_weights, h_weights, sizeof(h_weights));

    // Allocate device memory
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of N threads
    double s = cpusecond();
    stencil1D<<<1, N>>>(d_in, d_out);
   double e = cpusecond();
    // Copy result back
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Input : ");
    for (int i = 0; i < N; i++) printf("%d ", h_in[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) printf("%d ", h_out[i]);
    printf("\n");
printf("gpu computation time: %f\n",e-s);
    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
