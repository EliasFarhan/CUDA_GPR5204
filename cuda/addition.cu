#include <addition.h>
#include <cuda.h>

__global__
void add(int n, const float *x, const float *y, float* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		result[i] = x[i] + y[i];
}

void GpuAdd(const float* x, const float* y, float* result, const size_t N)
{
	float* gpuX;
	float* gpuY;
	float* gpuResult;
	cudaMallocManaged(&gpuX, N * sizeof(float));
	cudaMallocManaged(&gpuY, N * sizeof(float));

	cudaMallocManaged(&gpuResult, N * sizeof(float));


	cudaMemcpy(gpuX, x, N, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuY, y, N, cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add << <numBlocks, blockSize >> > (N, gpuX, gpuY, gpuResult);
	cudaDeviceSynchronize();
	cudaMemcpy(gpuResult, result, N, cudaMemcpyDeviceToHost);

	cudaFree(gpuX);
	cudaFree(gpuY);
	cudaFree(gpuResult);
}