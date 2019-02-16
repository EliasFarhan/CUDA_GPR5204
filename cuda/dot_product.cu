#include <cuda.h>
#include <dot.h>

#define BLOCK_SIZE 16

__global__ 
void dot(const float *a, const float *b, float *c, const int N)
{
	__shared__ float temp[BLOCK_SIZE];

	int index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if (index < N)
	{
		temp[threadIdx.x] = a[index] * b[index];
	}
	
	__syncthreads();

	if (threadIdx.x == 0)
	{
		float sum = 0.0f;
		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			if (i < N)
			{
				sum += temp[i];
			}
		}
		atomicAdd(c, sum);
	}
}

float dotGpu(const float *hostV1, const float *hostV2, size_t len)
{
	float *devV1, *devV2, *devResult;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&devV1, len * sizeof(float));
	cudaMallocManaged(&devV2, len * sizeof(float));
	cudaMallocManaged(&devResult, sizeof(float));
	*devResult = 0.0f;

	cudaMemcpy(devV1, hostV1, len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devV2, hostV2, len * sizeof(float), cudaMemcpyHostToDevice);


	const int numBlocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dot <<<numBlocks, BLOCK_SIZE >>> (devV1, devV2, devResult, len);
	cudaDeviceSynchronize();

	float v = *devResult;

	cudaFree(devV1);
	cudaFree(devV2);
	cudaFree(devResult);
	return v;
}

void matrixMultGpu(const float* m1, const float* m2, float* result, const size_t newWidth, const size_t newHeight, const size_t len)
{
	float *devV1, *devV2, *devResult;
	cudaMallocManaged(&devV1, len * newHeight * sizeof(float));
	cudaMallocManaged(&devV2, len * newWidth * sizeof(float));
	cudaMallocManaged(&devResult, newWidth*newHeight*sizeof(float));
	for (int i = 0; i < newWidth*newHeight; i++)
	{
		devResult[i] = 0.0f;
	}

	cudaMemcpy(devV1, m1, len * newHeight * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devV2, m2, len * newWidth * sizeof(float), cudaMemcpyHostToDevice);


	const int numBlocks = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	for (int y = 0; y < newHeight; y++)
	{
		for (int x = 0; x < newWidth; x++)
		{
			const int xOrigin = x * len;
			const int yOrigin = y * len;
			dot <<<numBlocks, BLOCK_SIZE >>> (devV1+yOrigin, devV2+xOrigin, devResult+(x*newWidth + y), len);
		}
	}

	cudaDeviceSynchronize();
	for (int i = 0; i < newWidth*newHeight; i++)
	{
		result[i] = devResult[i];
	}
	cudaFree(devV1);
	cudaFree(devV2);
	cudaFree(devResult);
}