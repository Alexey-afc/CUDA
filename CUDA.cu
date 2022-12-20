
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>
#define N 1000
using namespace std;

__global__ void mattmult(double* a, double* b, double* c)
{
	//глобальные координаты
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	//локальные
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	double sum = 0;
	int end = ((int)N / 32) + 1;
	for (int i = 0;i < end;i++)
	{
		__syncthreads();
		__shared__ double casheA[32][32];
		__shared__ double casheB[32][32];
		casheA[tx][ty] = a[x + N * ty + 32 * i * N];
		casheB[tx][ty] = b[tx + y * N + 32 * i];
		__syncthreads();
		for (int k = 0; k < 32; k++)
			sum += casheA[tx][k] * casheB[k][ty];
		
	}
	if (x < N && y < N)
		c[x + y * N] = sum;
}


int main()
{
	unsigned int mem_size = sizeof(double) * N * N;
	double* hA, * hB, * hC;
	double* da, * db, * dc;
	int threadsPerBlock = 32;
	int blocksPerGrid = int((N / ((float)threadsPerBlock)) + 1);
	dim3 blockDim(threadsPerBlock, threadsPerBlock, 1);
	dim3 gridDim(blocksPerGrid, blocksPerGrid, 1);
	cudaHostAlloc((void**)&hA, mem_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hB, mem_size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&hC, mem_size, cudaHostAllocDefault);

	cudaMalloc((void**)&da, mem_size);
	cudaMalloc((void**)&db, mem_size);
	cudaMalloc((void**)&dc, mem_size);

	for (int i = 0;i <= (N * N - 1);i++)
	{
		hA[i] = rand() % 100;
		hB[i] = rand() % 100;
		hC[i] = 0;
	}
	cudaStream_t stream[2];
	for (int i = 0; i < 2; ++i) cudaStreamCreate(&stream[i]);

	cudaEvent_t start, stop;
	float gpu_time = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//cudaMemcpy(da, hA, mem_size , cudaMemcpyHostToDevice);
	//cudaMemcpy(db, hB, mem_size , cudaMemcpyHostToDevice);

	cudaMemcpyAsync(da, hA, mem_size, cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(db, hB, mem_size, cudaMemcpyHostToDevice, stream[1]);

	cudaDeviceSynchronize();
	mattmult << < gridDim, blockDim>> > (da, db, dc);
	for (int i = 0; i < 2; i++)
	{
		int offset = int(N * N / 2);
		cudaMemcpyAsync(hC + offset * i, dc + offset * i, mem_size / 2, cudaMemcpyDeviceToHost, stream[i]);
	}
	//cudaMemcpy(hC, dc, mem_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_time, start, stop);
	cout << "Time GPU=" << gpu_time << " ms";
	for (int i = 0; i < 2; ++i) cudaStreamDestroy(stream[i]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout << "difference =" ;
	int dif = 0;
	for (int i =0 ;i <N;i++)
	{
		for (int j =0 ;j <N ;j++)
		{
			double s = 0;
			for (int k = 0;k < N;k++)
			{
				s += hA[i + k * N] * hB[k + j * N];
			}	
			dif += abs(s - hC[i + j * N]);
		}
	}
	cout  << dif << endl;
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);
	/*printf("\nYour matrix A=");
	for (int i = 0;i < N;i++) {
		printf("\n");
		for (int j = 0;j < N;j++) {
			printf("%0.2lf ", hA[i + j*N]);
		}
	}
	printf("\nYour matrix B=");
	for (int i = 0;i < N;i++) {
		printf("\n");
		for (int j = 0;j < N;j++) {
			printf("%0.2lf ", hB[i + j * N]);
		}
	}
	printf("\nYour matrix C=");
		for (int i = 0;i < N;i++) {
			printf("\n");
			for (int j = 0;j < N;j++) {
				printf("%0.2lf ", hC[i + j * N]);
			}
		}
	*/

}




