#include <cuda_runtime.h>

#define BLOCK_WIDTH 256;
__constant__ float kernelC[1024];


// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {

    int outputRows = input_rows + kernel_rows - 1;
    int outputCols = output_cols + kernel_rows - 1;

    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim((outputCols + BLOCK_WIDTH - 1) /BLOCK_WIDTH, (outputRows + BLOCK_WIDTH - 1) /BLOCK_WIDTH);

    cudaMemcpyToSymbol(kernelC, kernel, kernel_cols * kernel_rows * sizeof(float));

    sharedMemCols = BLOCK_WIDTH + kernel_cols - 1;
    sharedMemRows = BLOCK_WIDTH + kernel_rows - 1;
    size_t sharedMemSize = sharedMemCols * sharedMemRow * sizeof(float);

    convolution<<<gridDim, blockDim sharedMemSize>>>(input, kernel, output, input_rows,
        input_cols, kernel_rows, kernel_cols, outputRows, outputCols);
