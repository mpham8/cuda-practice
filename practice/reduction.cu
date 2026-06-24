#include <cuda_runtime.h>

#define LOADS_PER_THREAD 4
#define THREADS_PER_BLOCK 256 //needs multiple of 32
#define SHARED_MEM_SIZE (THREADS_PER_BLOCK/32)

__global__ void reduction(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tIdx = threadIdx.x;
    int warpIdx = tIdx / 32;
    int laneIdx = tIdx % 32;
    
    __shared__ float data_s[SHARED_MEM_SIZE];
    
    //LOAD MEM FROM GLOBAL MEM, STRIDED
    float sum = 0.0f;
    #pragma unroll
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) sum += input[i];

    //W/IN BLOCK REDUCTION
    //warp reduction (256 -> 8 per block)
    #pragma unroll
    for (int i = 16; i > 0; i>>=1) sum += __shfl_down_sync(0xffffffff, sum, i);

    //save warp reduction results to shared mem, lane 0 only
    if (laneIdx == 0) data_s[warpIdx] = sum;
    __syncthreads();

    //REDUCTION TO 1 PER BLOCK AND GLOBAL BLOCK REDUCTION
    //warp 0 executes block reduction
    if (warpIdx == 0) {
        sum = (laneIdx < SHARED_MEM_SIZE) ? data_s[laneIdx] : 0.0f;
        
        //warp reduction (8->1 per block)
        #pragma unroll
        for (int i = 16; i > 0; i>>=1) sum += __shfl_down_sync(0xffffffff, sum, i);
        //atomic add, lane 0 only (blocksPerGrid -> 1 globally)
        if (laneIdx == 0) atomicAdd(output, sum);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // int THREADS_PER_BLOCK = 256;
    int loadsPerBlock = LOADS_PER_THREAD * THREADS_PER_BLOCK; 
    int blocksPerGrid = ( N + loadsPerBlock - 1 )/loadsPerBlock;

    reduction<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, output, N);
}
