#include <cuda_runtime.h>
#include <limits>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int TILE_SIZE = 2048;


__global__ void softmaxAggTile(const float* input, float* output, float* maxD, float* sumD, int N){
    //*each block aggregates across 1 tile
    int tid = threadIdx.x;
    int bid = blockIdx.x * blockDim.x;

    //shared mem
    __shared__ float maxS = [THREADS_PER_BLOCK];
    __shared__ float sumS = [THREADS_PER_BLOCK];

    float maxV = 0.0f;
    float sumV = 0.0f;
    //*strided local accumulation per thread from TILE_SIZE: THREADS_PER_BLOCK
    for (int i = bid + tid; i < min(N, bid + TILE_SIZE); i += blockDim.x){
        float val = input[i];
        float newMaxV = fmaxf(maxV, val);
        sumV = __expf(maxV - newMaxV) * sumV + __expf(val - newMaxV);
        maxV = newMaxV;
    }
    maxS[tid] = maxV;
    sumS[tid] = sumV;
    __syncthreads();



    //*tree reduction using shared mem from TILE_SIZE : 32
    for (int i = THREADS_PER_BLOCK/2; i >= 32; i/=2){
        if (tid < i){
            float max1 = maxS[tid];
            float max2 = maxS[tid + i];
            float sum1 = sumS[tid];
            float sum2 = sumS[tid + i];
            
            float newMaxV = fmaxf(max1, max2);
            sumS[tid] = __expf(max1 - newMaxV) * sum1 + __expf(max2 - newMaxV) * sum_val2;
            sumS[tid] = newMaxV;

        }

        __syncthreads();
    }

    //*reduction using warp registers from 32 on

    //put in registers for warp reduction
    float maxR = maxS[tid];
    float sumR = sumS[tid];
    softmaxAggWarp(maxR, sumR);

    if (threadIdx.x == 0){
        maxD[blockIdx.x] = maxR;
        sumD[blockIdx.x] = sumR;
    }

}


extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + TILE_SIZE - 1)/TILE_SIZE;
    
    float* maxD = nullptr;
    float* sumD = nullptr;
    cudaMalloc(&maxD, sizeof(float) * blocksPerGrid);
    cudaMalloc(&sumD, sizeof(float) * blocksPerGrid);

    softmaxAggTile<<< blocksPerGrid, threadsPerBlock>>>(input, output, maxD, sumD, N);

    cudaFree(maxD);
    cudaFree(sumD);
}