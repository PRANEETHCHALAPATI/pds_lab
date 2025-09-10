#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

double cpusecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void dotproductkernel(int* a, int* b, int* result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = tid;

    // Step 1: Element-wise multiplication
    if (index < n) {
        a[index] = a[index] * b[index];
    }
    __syncthreads();

    // Step 2: Parallel reduction
    for (int stride = 1; stride < n; stride *= 2) {
        if (index % (2 * stride) == 0 && (index + stride) < n) {
            a[index] += a[index + stride];
        }
        __syncthreads();
    }

    // Step 3: Write result
    if (index == 0) {
        *result = a[0];
    }
}

int main() {
    const int N = 5;

    int h_a[N] = {1, 2, 3, 4, 5};
    int h_b[N] = {10, 20, 30, 40, 50};
    int h_result = 0;

    int *d_a, *d_b, *d_result;

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    double gpu_start = cpusecond();

    int threadsperblock = 5;
    int blocks = (N + threadsperblock - 1) / threadsperblock;

    dotproductkernel<<<blocks, threadsperblock>>>(d_a, d_b, d_result, N);
    cudaDeviceSynchronize();

    double gpu_end = cpusecond();

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Dot product: %d\n", h_result);
    printf("GPU computation time: %f seconds\n", gpu_end - gpu_start);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
