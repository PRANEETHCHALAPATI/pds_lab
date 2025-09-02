#include<stdio.h>
#include<sys/time.h>

double cpuSecond() {
		
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void dot_product_kernel(int *a,int *b,int *result,int n) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int index = tid;
	
	if(index < n) {
		
		a[index] = a[index] * b[index];
	}
	__syncthreads();
	
	for(int stride=1; stride<n; stride*=2) {
		
		__syncthreads();
		if (index%(2*stride) == 0 && (index + stride)<n) {

			a[index] += a[index+stride];
		}
	}

	if(index == 0) {

		*result = a[0];
	}

}

int main(void) {

	const int N = 5;
	int h_a[5] = {1,2,3,4,5};
	int h_b[5] = {10,20,30,40,50};
	int h_result = 0;
	
	int *d_a,*d_b,*d_result;
	
	cudaMalloc((void**)&d_a,N*sizeof(int));
	cudaMalloc((void**)&d_b,N*sizeof(int));
	cudaMalloc((void**)&d_result,sizeof(int));
	
	cudaMemcpy(d_a,h_a,N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,N*sizeof(int),cudaMemcpyHostToDevice);
	
	int threadsPerBlock;
	printf("enter the no of threads to be used per block : ");
	scanf("%d",&threadsPerBlock);
	int blocks = (N + threadsPerBlock-1)/threadsPerBlock;
	double start_time = cpuSecond();
	dot_product_kernel<<<blocks,threadsPerBlock>>>(d_a,d_b,d_result,N);
	cudaDeviceSynchronize();
	double end_time = cpuSecond();
	cudaMemcpy(&h_result,d_result,sizeof(int),cudaMemcpyDeviceToHost);
	printf("dot product result : %d\n",h_result);
	printf("Gpu computation time : %f seconds.\n",end_time - start_time);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	return 0;
}
