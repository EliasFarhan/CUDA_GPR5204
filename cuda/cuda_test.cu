#include <cstdio>
__global__
void print_values()
{
    printf("Block Idx: %d, BlockDim: %d, Thread Idx: %d, GridDim : %d\n",blockIdx.x,blockDim.x,threadIdx.x, gridDim.x);
}

void test_cuda(int numBlocks, int blockSize)
{
    print_values<< <numBlocks, blockSize >> > ();
    cudaDeviceSynchronize();
}