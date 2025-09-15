#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<sys/time.h>
#include <cuda_runtime.h>

#define N 3 // 3x3 matrix
double cpusecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void transposer(int *input, int *output, int width) { 
    __shared__ int tile[N][N]; // Shared memory tile for 3x3

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * width) { // Prevent out-of-bounds
        int row = idx / width;  // Convert 1D thread index -> 2D coords
        int col = idx % width;

        // Load into shared memory (transposed)
        tile[col][row] = input[row * width + col];

        __syncthreads(); // Wait for all threads

        // Write back transposed value
        output[idx] = tile[row][col];
    }
}

int main() {
    int size = N * N * sizeof(int);
    int h_input[N * N], h_output[N * N];

    // Generate random input matrix
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_input[i] = rand() % 100;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    
    double s = cpusecond();

    transposer<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    
    double e = cpusecond();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Display original matrix
    printf("Original Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%2d ", h_input[i * N + j]);
        }
        printf("\n");
    }

    // Display transposed matrix
    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%2d ", h_output[i * N + j]);
        }
        printf("\n");
    }
    
    printf("gpu computation time: %f\n",e-s);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
