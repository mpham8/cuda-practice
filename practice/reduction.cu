#include <cuda_runtime.h>

#define LOADS_PER_THREAD 4
#define threadsPerBlock 256 //needs multiple 32
#define SHARED_MEM_SIZE (threadsPerBlock/32)

__device__ float warpReduce(float val){
    #pragma unroll
    for (int i = 16; i > 0; i>>=1) val += __shfl_down_sync(0xffffffff, val, i);
    return val;
    }


__global__ void reduction(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tIdx = threadIdx.x;
    int warpIdx = tIdx / 32;
    int laneIdx = tIdx % 32;
    
    __shared__ float data_s[SHARED_MEM_SIZE];
    
    //LOAD MEM FROM GLOBAL MEM
    float sum = 0.0f;
    #pragma unroll
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) sum += input[i];
    // data_s[tIdx] = sum;
    // __syncthreads();

    
    //REDUCE W/IN BLOCK
    // for (int i = blockDim.x; i > 32; i >>= 1){
    //     if (tIdx < i) data_s[tIdx] += data_s[tIdx + i];
    //     __syncthreads();
    // }
    
    //warp reduction 256->8
    sum = warpReduce(sum);

    if (laneIdx == 0) data_s[warpIdx] = sum;
    __syncthreads();

    //REDUCE TO 1 PER BLOCK AND GLOBALLY AMONG BLOCKS
    if (warpIdx == 0) {
        sum = (laneIdx < SHARED_MEM_SIZE) ? data_s[laneIdx] : 0.0f;
        sum = warpReduce(sum);
        //atomic add
        if (laneIdx == 0) atomicAdd(output, sum);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // int threadsPerBlock = 256;
    int loadsPerBlock = LOADS_PER_THREAD * threadsPerBlock; 
    int blocksPerGrid = ( N + loadsPerBlock - 1 )/loadsPerBlock;

    reduction<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
