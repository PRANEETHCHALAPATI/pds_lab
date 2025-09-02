
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define N (1 << 5) // 1,048,576 elements
#define THREADS_PER_BLOCK 256

// Kernel for Odd-Even Transposition Sort using global memory
__global__ void odd_even_sort_kernel(int* arr, int n, int phase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (phase % 2 == 0) { // Even phase
        // Even-indexed threads compare (i, i+1) pairs
        if (idx % 2 == 0 && idx < n - 1) {
            if (arr[idx] > arr[idx + 1]) {
                // Swap elements
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
    } else { // Odd phase
        // Odd-indexed threads compare (i, i+1) pairs
        if (idx % 2 == 1 && idx < n - 1) {
            if (arr[idx] > arr[idx + 1]) {
                // Swap elements
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
    }
}

// Kernel using shared memory for better performance
__global__ void odd_even_sort_shared_kernel(int* arr, int n, int phase) {
    extern __shared__ int shared_arr[];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory
    if (idx < n) {
        shared_arr[tid] = arr[idx];
    }
    __syncthreads();
    
    if (phase % 2 == 0) { // Even phase
        if (tid % 2 == 0 && tid < blockDim.x - 1 && (idx + 1) < n) {
            if (shared_arr[tid] > shared_arr[tid + 1]) {
                int temp = shared_arr[tid];
                shared_arr[tid] = shared_arr[tid + 1];
                shared_arr[tid + 1] = temp;
            }
        }
    } else { // Odd phase
        if (tid % 2 == 1 && tid < blockDim.x - 1 && (idx + 1) < n) {
            if (shared_arr[tid] > shared_arr[tid + 1]) {
                int temp = shared_arr[tid];
                shared_arr[tid] = shared_arr[tid + 1];
                shared_arr[tid + 1] = temp;
            }
        }
    }
    __syncthreads();
    
    // Write back to global memory
    if (idx < n) {
        arr[idx] = shared_arr[tid];
    }
}

// Initialize array with random values
void initialize_array(int* arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 10000 + 1; // Random numbers between 1-10000
    }
}

// CPU sorting function for validation
void cpu_sort(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Check if array is sorted
int is_sorted(int* arr, int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            return 0;
        }
    }
    return 1;
}

// Validate GPU results against CPU results
int validate_results(int* cpu_arr, int* gpu_arr, int size) {
    for (int i = 0; i < size; i++) {
        if (cpu_arr[i] != gpu_arr[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpu_arr[i], gpu_arr[i]);
            return 0;
        }
    }
    return 1;
}


// ------------------------ Main Function ------------------------
int main() {
    int *h_arr, *h_cpu_sorted, *h_gpu_result;
    int *d_arr;

    size_t size = N * sizeof(int);
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // ------------------ Measure GPU memory before allocation ------------------
    size_t free_mem_before, total_mem;
    cudaMemGetInfo(&free_mem_before, &total_mem);
    printf("\n=== GPU Memory Info ===\n");
    printf("Before allocation: Free GPU memory: %.2f MB / %.2f MB\n",
           free_mem_before / (1024.0 * 1024),
           total_mem / (1024.0 * 1024));

    // ------------------ Allocate host memory ------------------
    h_arr = (int*)malloc(size);
    h_cpu_sorted = (int*)malloc(size);
    h_gpu_result = (int*)malloc(size);

    // ------------------ Initialize and copy arrays ------------------
    initialize_array(h_arr, N);
    memcpy(h_cpu_sorted, h_arr, size);
    memcpy(h_gpu_result, h_arr, size);

    // ------------------ CPU Sort ------------------
    clock_t cpu_start = clock();
    cpu_sort(h_cpu_sorted, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

    // ------------------ Allocate device memory ------------------
    cudaMalloc(&d_arr, size);

    // ------------------ Measure GPU memory after allocation ------------------
    size_t free_mem_after;
    cudaMemGetInfo(&free_mem_after, &total_mem);
    printf("After allocation: Free GPU memory: %.2f MB\n",
           free_mem_after / (1024.0 * 1024));
    printf("Approx GPU memory used: %.2f MB\n",
           (free_mem_before - free_mem_after) / (1024.0 * 1024));

    // ------------------ Copy data to device ------------------
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // ------------------ Setup CUDA timing ------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ------------------ GPU Sort ------------------
    for (int phase = 0; phase < N; phase++) {
        odd_even_sort_shared_kernel<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(
            d_arr, N, phase);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // ------------------ Measure GPU execution time ------------------
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_time = gpu_ms / 1000.0;

    // ------------------ Copy results back and validate ------------------
    cudaMemcpy(h_gpu_result, d_arr, size, cudaMemcpyDeviceToHost);
    int is_valid = validate_results(h_cpu_sorted, h_gpu_result, N);
    int gpu_sorted = is_sorted(h_gpu_result, N);
    int cpu_sorted = is_sorted(h_cpu_sorted, N);

    // ------------------ Print Final Report ------------------
    printf("\n=== Sorting Report ===\n");
    printf("Array size: %d elements\n", N);
//  printf("Execution time (CPU): %.6f seconds\n", cpu_time);
    printf("Execution time (GPU): %.6f seconds\n", gpu_time);
    printf("Validation: %s\n", is_valid ? "PASSED" : "FAILED");
    printf("CPU array sorted: %s\n", cpu_sorted ? "YES" : "NO");
    printf("GPU array sorted: %s\n", gpu_sorted ? "YES" : "NO");

    if (cpu_time > 0) {
        printf("Speedup (CPU / GPU): %.2f x\n", cpu_time / gpu_time);
    }

    // ------------------ Cleanup ------------------
    free(h_arr);
    free(h_cpu_sorted);
    free(h_gpu_result);
    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
