#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
#define MAX_BLOCK_NUM 1024
__device__ float gBlockSum[MAX_BLOCK_NUM];

__device__ __forceinline__ float warpPrefix(float val, int laneIdx){
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1){
        float curr = __shfl_up_sync(0xffffffff, val, offset);
        if (laneIdx >= offset) val += curr;
    }
    return val;
}

__global__ void blockPrefix(const float* input, float* output, int N){
    //ids
    int tIdx = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tIdx;
    int laneIdx = tIdx % 32;
    int warpIdx = tIdx / 32;
    int warpPerBlock = THREAD_PER_BLOCK / 32;

    extern __shared__ float sMem[];

    // global load
    if (idx < N) {
        sMem[tIdx] = input[idx];
    } else {
        sMem[tIdx] = 0.0f;
    }
    __syncthreads();
    
    //warp reduction 256->8 prefixes per block
    float val = warpPrefix(sMem[tIdx], laneIdx);
    __syncthreads();
    sMem[tIdx] = val;
    __syncthreads();

    //8 -> 1 prefix per block
    for (int offset = 1; offset < warpPerBlock; offset *= 2){
        if (warpIdx >= offset){
            float curr = sMem[(warpIdx - offset) * 32 + 32 - 1];
            val += curr;
        }
        __syncthreads();
        if (warpIdx >= offset) sMem[tIdx] = val;
        __syncthreads();
    }

    //setup global reduction reduction
    if (idx < N) {
        output[idx] = sMem[tIdx];
    }
    if (tIdx == 0) {
        gBlockSum[blockIdx.x] = sMem[THREAD_PER_BLOCK - 1];
    }
}


__global__ void globalPrefix(float* output, int N) {
    int tIdx = threadIdx.x;
    int bIdx = blockIdx.x;
    int idx = bIdx * blockDim.x + tIdx;

    float curr = 0.0f;
    for (int i = tIdx; i < bIdx; i += blockDim.x) {
        curr += gBlockSum[i];
    }

    __shared__ float smem[THREAD_PER_BLOCK];
    smem[tIdx] = curr;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tIdx < s) smem[tIdx] += smem[tIdx + s];
        __syncthreads();
    }

    float carry = smem[0];
    if (idx < N) {
        output[idx] += carry;
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blocksPerGrid = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    size_t sharedMemBytes = THREAD_PER_BLOCK * sizeof(float);

    blockPrefix<<<blocksPerGrid, THREAD_PER_BLOCK, sharedMemBytes>>>(input, output, N);
    globalPrefix<<<blocksPerGrid, THREAD_PER_BLOCK>>>(output, N);
}
