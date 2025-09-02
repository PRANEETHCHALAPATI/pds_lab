#include<stdio.h>
#include<sys/time.h>
#define N 6

double cpusecond() {

	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void transposer(int* input,int* output,int width) {

	__shared__ int tile[N][N];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < width * width) {
		
		int row = idx/width;
		int col = idx % width;
		
		tile[col][row] = input[row*width + col];
		
		__syncthreads();
		
		output[idx] = tile[row][col];
	}
}

int main() {
	
	int size = N * N * sizeof(int);
	int h_input[N*N],h_output[N*N];
	
	srand(time(NULL));
	for (int i=0; i<N*N; i++) {

		h_input[i] = rand() % 100;
	}

	int *d_output,*d_input;
	cudaMalloc(&d_input,size);
	cudaMalloc(&d_output,size);
	
	cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice);
	
	int threadsPerBlock = N*N;
	int blocksPerGrid = (N*N + threadsPerBlock-1)/threadsPerBlock;
	double start_time = cpusecond();
	transposer<<<blocksPerGrid,threadsPerBlock>>>(d_input,d_output,N);
	cudaDeviceSynchronize();
	double end_time = cpusecond();
	cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost);
	printf("Original Matrix : \n");
	for (int i=0; i<N; i++) {

		for (int j=0; j<N; j++) {

			printf("%d\t",h_input[i*N+j]);
		}
		printf("\n\n");
	}
	
	printf("Transposed Matrix : \n");
	for (int i=0; i<N; i++) {

		for (int j=0; j<N; j++) {

			printf("%d\t",h_output[i*N+j]);
		}
		printf("\n\n");
	}

	printf("GPU elapsed time : %fseconds\n",end_time-start_time);
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}

	









































