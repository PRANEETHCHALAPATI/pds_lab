#include<stdio.h>
#include<sys/time.h>
#define N 8
#define RADIUS 2
double cpuSecond() {
		
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__constant__ int d_weights[2*RADIUS + 1];

__global__ void stencil1D(int *in,int *out) {

	int i = threadIdx.x;
	
	int result = 0;
	for(int j=-RADIUS; j<= RADIUS; j++) {

		int idx = i + j;
		if (idx >= 0 && idx < N) {

			result += d_weights[j + RADIUS] * in[idx];
		}
	}
	out[i] = result;
}
int main() {

	int h_in[N] = {1,2,3,4,5,6,7,8};
	int h_out[N];
	
	int *d_in,*d_out;
	
	int h_weights[2*RADIUS + 1] = {1,1,1,1,1};
	
	cudaMemcpyToSymbol(d_weights,h_weights,sizeof(h_weights));

	cudaMalloc(&d_in,N*sizeof(int));
	cudaMalloc(&d_out,N*sizeof(int));
	
	cudaMemcpy(d_in,h_in,N*sizeof(int),cudaMemcpyHostToDevice);
 
	double start_time = cpuSecond();
	stencil1D<<<1,N>>>(d_in,d_out);
  double end_time = cpuSecond();
	cudaMemcpy(h_out,d_out,N*sizeof(int),cudaMemcpyDeviceToHost);
	
	printf("input : ");
	for (int i=0; i<N; i++) printf("%d\t",h_in[i]);
	printf("\noutput : ");
	for (int i=0; i<N; i++) printf("%d\t",h_out[i]);
	printf("\n");
	printf("GPU computational time %f\n",end_time-start_time);
	cudaFree(d_in);
	cudaFree(d_out);
}	
