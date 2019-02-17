#include <cuda.h>
#include <matrix.h>

#define BLOCK_SIZE 256

__global__ void gpu_matrix_mult(const float *a, const float *b, float *c, int m, int n, int k)
{
    int row1 = blockIdx.y * blockDim.y + threadIdx.y;
    int row2 = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if( row2 < k && row1 < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row1 * n + i] * b[row2 * n + i];
        }
        c[row1 * k + row2] = sum;
    }
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

    const unsigned m = newHeight;
    const unsigned k = newWidth;
    const unsigned n = len;
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(devV1, devV2, devResult, m,n,k);

    cudaDeviceSynchronize();
    for (int i = 0; i < newWidth*newHeight; i++)
    {
        result[i] = devResult[i];
    }
    cudaFree(devV1);
    cudaFree(devV2);
    cudaFree(devResult);
}