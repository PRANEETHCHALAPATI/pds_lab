#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define INSERTION_SORT_THRESHOLD 32
#define STACK_SIZE 1024
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 64
#define N (1 << 10) // Change to 1<<12, 1<<14 for scaling tests

typedef struct {
    int low;
    int high;
} SubArray;

__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void insertion_sort(int* arr, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

__global__ void parallel_quicksort_kernel(int* arr, int n, SubArray* stack, int* stack_top, int* sorted_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        while (atomicAdd(sorted_count, 0) < n) {
            int pos = atomicSub(stack_top, 1) - 1;

            if (pos >= 0) {
                SubArray task = stack[pos];
                int low = task.low;
                int high = task.high;

                while (low < high) {
                    if ((high - low + 1) < INSERTION_SORT_THRESHOLD) {
                        insertion_sort(arr, low, high);
                        atomicAdd(sorted_count, (high - low + 1));
                        break;
                    }

                    int pivot = partition(arr, low, high);
                    atomicAdd(sorted_count, 1);

                    int left_size = pivot - low;
                    int right_size = high - pivot;

                    if (left_size > 0 && right_size > 0) {
                        if (left_size > right_size) {
                            int new_pos = atomicAdd(stack_top, 1);
                            if (new_pos < STACK_SIZE) {
                                stack[new_pos].low = low;
                                stack[new_pos].high = pivot - 1;
                            }
                            low = pivot + 1;
                        } else {
                            int new_pos = atomicAdd(stack_top, 1);
                            if (new_pos < STACK_SIZE) {
                                stack[new_pos].low = pivot + 1;
                                stack[new_pos].high = high;
                            }
                            high = pivot - 1;
                        }
                    } else if (left_size > 0) {
                        high = pivot - 1;
                    } else if (right_size > 0) {
                        low = pivot + 1;
                    } else {
                        break;
                    }
                }

                if (low == high) {
                    atomicAdd(sorted_count, 1);
                }
            } else {
                __threadfence();
            }
        }
    }
}

void initialize_data(int* arr, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 10000 + 1;
    }
}

int validate_sort(int* cpu_sorted, int* gpu_sorted, int size) {
    for (int i = 0; i < size; i++) {
        if (cpu_sorted[i] != gpu_sorted[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, cpu_sorted[i], gpu_sorted[i]);
            return 0;
        }
    }
    return 1;
}

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

int main() {
    int* h_arr = (int*)malloc(N * sizeof(int));
    int* h_cpu_sorted = (int*)malloc(N * sizeof(int));
    int* h_gpu_result = (int*)malloc(N * sizeof(int));

    initialize_data(h_arr, N);
    memcpy(h_cpu_sorted, h_arr, N * sizeof(int));
    memcpy(h_gpu_result, h_arr, N * sizeof(int));

    // Measure CPU execution time
    clock_t cpu_start = clock();
    cpu_sort(h_cpu_sorted, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // GPU memory tracking before allocation
    size_t free_mem_before, total_mem;
    cudaMemGetInfo(&free_mem_before, &total_mem);
    printf("\n=== GPU Memory Info ===\n");
    printf("Free memory before allocation: %.2f MB\n", free_mem_before / (1024.0 * 1024));

    // Allocate GPU memory
    int* d_arr;
    SubArray* d_stack;
    int* d_stack_top;
    int* d_sorted_count;
    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMalloc((void**)&d_stack, STACK_SIZE * sizeof(SubArray));
    cudaMalloc((void**)&d_stack_top, sizeof(int));
    cudaMalloc((void**)&d_sorted_count, sizeof(int));

    // GPU memory tracking after allocation
    size_t free_mem_after;
    cudaMemGetInfo(&free_mem_after, &total_mem);
    printf("Free memory after allocation: %.2f MB\n", free_mem_after / (1024.0 * 1024));
    printf("Approximate memory used: %.2f MB\n", 
           (free_mem_before - free_mem_after) / (1024.0 * 1024));

    int h_stack_top = 0;
    int h_sorted_count = 0;

    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack_top, &h_stack_top, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_count, &h_sorted_count, sizeof(int), cudaMemcpyHostToDevice);

    SubArray initial_task = {0, N - 1};
    cudaMemcpy(d_stack, &initial_task, sizeof(SubArray), cudaMemcpyHostToDevice);
    h_stack_top = 1;
    cudaMemcpy(d_stack_top, &h_stack_top, sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;

    // GPU timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    parallel_quicksort_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, N, d_stack, d_stack_top, d_sorted_count);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_time = gpu_ms / 1000.0;

    // Copy result back
    cudaMemcpy(h_gpu_result, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Validate and print
    printf("\n=== Sort Validation ===\n");
    if (validate_sort(h_cpu_sorted, h_gpu_result, N)) {
        printf("Sort validation: PASSED\n");
    } else {
        printf("Sort validation: FAILED\n");
    }

    printf("\n=== Timing ===\n");
//  printf("CPU time: %.6f seconds\n", cpu_time);
    printf("GPU time: %.6f seconds\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup (CPU / GPU): %.2fx\n", cpu_time / gpu_time);
    }

    // Optional preview
    printf("\nFirst 10 CPU sorted elements: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_cpu_sorted[i]);
    printf("\n");

    printf("First 10 GPU sorted elements: ");
    for (int i = 0; i < 10; i++) printf("%d ", h_gpu_result[i]);
    printf("\n");

    // Cleanup
    free(h_arr);
    free(h_cpu_sorted);
    free(h_gpu_result);
    cudaFree(d_arr);
    cudaFree(d_stack);
    cudaFree(d_stack_top);
    cudaFree(d_sorted_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
