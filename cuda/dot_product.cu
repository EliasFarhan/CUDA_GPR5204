#include <cuda.h>
#include <dot.h>

#define BLOCK_SIZE 512

__global__ void dot(const float *a, const float *b, float *c, const int N)
{
	__shared__ float temp[BLOCK_SIZE];
	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		if (index < BLOCK_SIZE)
		{
			temp[index] = a[index] * b[index];
		}
		else
		{
			temp[index % BLOCK_SIZE] += a[index] * b[index];
		}
	}
	
	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sum = 0;
		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			sum += temp[i];
		}
		c[0] = sum;
	}
}


float dotGpu(const float *hostV1, const float *hostV2, size_t len)
{
	float *devV1, *devV2, *devResult;
	float result = 0.0f;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&devV1, len * sizeof(float));
	cudaMallocManaged(&devV2, len * sizeof(float));

	cudaMallocManaged(&devResult, sizeof(float));

	cudaMemcpy(devV1, hostV1, len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devV2, hostV2, len * sizeof(float), cudaMemcpyHostToDevice);


	int numBlocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dot << <numBlocks, BLOCK_SIZE >> > (devV1, devV2, devResult, len);
	cudaDeviceSynchronize();
	cudaMemcpy(devResult, &result, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(devV1);
	cudaFree(devV2);
	cudaFree(devResult);
	return result;
}