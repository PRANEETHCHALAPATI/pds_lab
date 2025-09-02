#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for odd-even sort
_global_ void oddEvenSortKernel(int *d_array, int n, int phase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n - 1) {
        if ((phase % 2 == 0 && idx % 2 == 0) || 
            (phase % 2 == 1 && idx % 2 == 1)) {
            if (d_array[idx] > d_array[idx + 1]) {
                int temp = d_array[idx];
                d_array[idx] = d_array[idx + 1];
                d_array[idx + 1] = temp;
            }
        }
    }
}

// Host function for sorting
void oddEvenSort(int *h_array, int n) {
    int *d_array;
    size_t size = n * sizeof(int);

    cudaMalloc(&d_array, size);
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int phase = 0; phase < n; phase++) {
        oddEvenSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, n, phase);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

// Utility function
void printArray(int *array, int n) {
    for (int i = 0; i < n; i++) printf("%d ", array[i]);
    printf("\n");
}

int main() {
    int n;
    printf("Enter array size: ");
    scanf("%d", &n);

    int h_array = (int)malloc(n * sizeof(int));
    printf("Enter %d numbers:\n", n);
    for (int i = 0; i < n; i++) scanf("%d", &h_array[i]);

    printf("\nOriginal: ");
    printArray(h_array, n);

    oddEvenSort(h_array, n);

    printf("Sorted:   ");
    printArray(h_array, n);

    free(h_array);
    return 0;
}