#include <cuda_runtime.h>
#include <limits>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int TILE_SIZE = 2048;


__device__ __forceinline__ void softmaxAggWarp(float& maxR, double& sumR){
    unsigned int warpMask = 0xffff'ffffu;
    for (int i = 16; i > 0; i/=2){
        float maxR2 = __shfl_down_sync(warpMask, maxR, i);
        double sumR2 = __shfl_down_sync(warpMask, sumR, i);
        
        float newMaxV = fmaxf(maxR, maxR2);
        sumR = exp((double)(maxR - newMaxV)) * sumR + exp((double)(maxR2 - newMaxV)) * sumR2;
        maxR = newMaxV;
    }
}


__global__ void softmaxAggGlobal(const float* input, float* output, float* maxD, double* sumD, int N){
    //*each block aggregates across 1 tile
    int tid = threadIdx.x;
    int bid = blockIdx.x * TILE_SIZE;

    //shared mem
    __shared__ float maxS[THREADS_PER_BLOCK];
    __shared__ double sumS[THREADS_PER_BLOCK];

    float maxV = std::numeric_limits<float>::lowest();
    double sumV = 0.0;

    //*aggregate across tiles globally
    // int blocksPerGrid = (N + TILE_SIZE - 1)/TILE_SIZE;
    for (int i = tid; i < gridDim.x; i += blockDim.x){
        float max2 = maxD[i];
        double sum2 = sumD[i];

        float newMaxV = fmaxf(maxV, max2);
        sumV = exp((double)(maxV - newMaxV)) * sumV + exp((double)(max2 - newMaxV)) * sum2;
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
            double sum1 = sumS[tid];
            double sum2 = sumS[tid + i];
            
            float newMaxV = fmaxf(max1, max2);
            sumS[tid] = exp((double)(max1 - newMaxV)) * sum1 + exp((double)(max2 - newMaxV)) * sum2;
            maxS[tid] = newMaxV;

        }

        __syncthreads();
    }

    //*reduction using warp registers from 32 on
    //put in registers for warp reduction
    double sumR = 0.0;
    float maxR = -INFINITY;
    if (tid < 32){
        maxR = maxS[tid];
        sumR = sumS[tid];
        softmaxAggWarp(maxR, sumR);
    }
    if (tid == 0) {
        maxS[0] = maxR;
        sumS[0] = sumR;
    }
    __syncthreads();

    //0th element had the softmaxAggWarp agg
    float maxG = maxS[0];
    double sumG = sumS[0];

    //*calculate per element softmax
    for (int i = bid + tid; i < min(N, bid + TILE_SIZE); i += blockDim.x){
        float val = input[i];
        output[i] = (float)(exp((double)(val - maxG)) / sumG);
    }
}


__global__ void softmaxAggTile(const float* input, float* output, float* maxD, double* sumD, int N){
    //*each block aggregates across 1 tile
    int tid = threadIdx.x;
    int bid = blockIdx.x * TILE_SIZE;

    //shared mem
    __shared__ float maxS[THREADS_PER_BLOCK];
    __shared__ double sumS[THREADS_PER_BLOCK];

    float maxV = std::numeric_limits<float>::lowest();
    double sumV = 0.0;
    //*strided local accumulation per thread from TILE_SIZE: THREADS_PER_BLOCK
    for (int i = bid + tid; i < min(N, bid + TILE_SIZE); i += blockDim.x){
        float val = input[i];
        float newMaxV = fmaxf(maxV, val);
        sumV = exp((double)(maxV - newMaxV)) * sumV + exp((double)(val - newMaxV));
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
            double sum1 = sumS[tid];
            double sum2 = sumS[tid + i];
            
            float newMaxV = fmaxf(max1, max2);
            sumS[tid] = exp((double)(max1 - newMaxV)) * sum1 + exp((double)(max2 - newMaxV)) * sum2;
            maxS[tid] = newMaxV;

        }

        __syncthreads();
    }

    //*reduction using warp registers from 32 on
    //put in registers for warp reduction
    double sumR = 0.0;
    float maxR = std::numeric_limits<float>::lowest();
    if (tid < 32){
        maxR = maxS[tid];
        sumR = sumS[tid];
        softmaxAggWarp(maxR, sumR);
    }

    if (threadIdx.x == 0){
        maxD[blockIdx.x] = maxR;
        sumD[blockIdx.x] = sumR;
    }

}


extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + TILE_SIZE - 1)/TILE_SIZE;
    
    float* maxD = nullptr;
    double* sumD = nullptr;
    cudaMalloc(&maxD, sizeof(float) * blocksPerGrid);
    cudaMalloc(&sumD, sizeof(double) * blocksPerGrid);

    softmaxAggTile<<< blocksPerGrid, threadsPerBlock>>>(input, output, maxD, sumD, N);
    
    softmaxAggGlobal<<< blocksPerGrid, threadsPerBlock>>>(input, output, maxD, sumD, N);

    cudaFree(maxD);
    cudaFree(sumD);
}