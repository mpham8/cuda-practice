#include <cuda_runtime.h>

#define BLOCK_WIDTH 32
__constant__ float kernelC[1024];

__global__ void convolution(const float* input, const float* kernel, float* output, int inputRows,
    int inputCols, int kernelRows, int kernelCols, int outputRows, int outputCols) {

    extern __shared__ float tileS[];

    int tI = threadIdx.y;
    int tJ = threadIdx.x;
    int blockI = blockIdx.y * BLOCK_WIDTH;
    int blockJ = blockIdx.x * BLOCK_WIDTH;
    int tileH = BLOCK_WIDTH + kernelRows - 1;
    int tileW = BLOCK_WIDTH + kernelCols - 1;

    //load from global to shared
    for (int i = tI; i < tileH; i += BLOCK_WIDTH){
        for (int j = tJ; j < tileW; j += BLOCK_WIDTH){
            int gI = blockI + i;
            int gJ = blockJ + j;
            tileS[i * tileW + j] = (gI < inputRows && gJ < inputCols) ? input[gI * inputCols + gJ] : 0.0f;
        }
    }
    __syncthreads();

    //kernel mul
    float cum = 0.0f;
    if ((blockI + tI) < outputRows && (blockJ + tJ) < outputCols) {
        for (int i = 0; i < kernelRows; i++) {
            for (int j = 0; j < kernelCols; j++) {
                cum += tileS[(tI + i) * tileW + (tJ + j)] * kernelC[i * kernelCols + j];
            }
        }
        //store
        output[(blockI + tI) * outputCols + (blockJ + tJ)] = cum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    
    //calculate output dims
    int outputRows = input_rows - kernel_rows + 1;
    int outputCols = input_cols - kernel_cols + 1;
    
    //calculate block, grid dims
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim((outputCols + BLOCK_WIDTH - 1) /BLOCK_WIDTH, (outputRows + BLOCK_WIDTH - 1) /BLOCK_WIDTH);
    
    //move over kernel
    cudaMemcpyToSymbol(kernelC, kernel, kernel_cols * kernel_rows * sizeof(float));

    //calculate shared memory size
    int sharedMemCols = BLOCK_WIDTH + kernel_cols - 1;
    int sharedMemRows = BLOCK_WIDTH + kernel_rows - 1;
    size_t sharedMemSize = sharedMemCols * sharedMemRows * sizeof(float);

    convolution<<<gridDim, blockDim, sharedMemSize>>>(input, kernel, output, input_rows,
        input_cols, kernel_rows, kernel_cols, outputRows, outputCols);

}
