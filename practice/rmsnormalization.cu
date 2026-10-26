#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256 //needs multiple of 32
#define SHARED_MEM_SIZE (THREADS_PER_BLOCK/32)

__device__ float sumSqGlobal = 0.0f;
//sum of warp via warp reduction
__device__ __forceinline__ float warpSum(float sum){
    #pragma unroll
    for (int i=16; i>0; i>>=1){
        sum += __shfl_down_sync(0xffffffff, sum, i);
    }
    return sum;
}


//recast float ptr as float4 struct ptr
__device__ __forceinline__ float4 loadFloat4(const float* ptr){
    return reinterpret_cast<const float4*>(ptr)[0];
}


//store float4 struct into float
__device__ __forceinline__ void storeFloat4(float* ptr, const float4& v){
    reinterpret_cast<float4*>(ptr)[0] = v;
}


//reduction to get root mean square
__global__ void rmsReduction(const float* input, const float gamma, const float beta, float* output, const int N,
    const float eps){
    int tIdx = threadIdx.x;
    int bIdx = blockIdx.x * blockDim.x;
    int idx = tIdx + bIdx;
    int warpIdx = tIdx / 32;
    int laneIdx = tIdx % 32;

    int numVec = N/4;
    int stride = blockDim.x * gridDim.x;

    __shared__ float data_s[SHARED_MEM_SIZE];

    //strided access
    float sum = 0.0f;
    for (int i = idx; i < numVec; i += stride) {
        float4 v = loadFloat4(input + 4 * i);
        sum += (v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w);
    }
    for (int i = 4 * numVec + idx; i < N; i += stride) {
        float v = input[i];
        sum += v * v;
    }


    //warp reduction 256 -> 8 per block
    sum = warpSum(sum);
    if (laneIdx == 0) data_s[warpIdx] = sum;
    __syncthreads();

    if (warpIdx == 0) {
        sum = (laneIdx < SHARED_MEM_SIZE) ? data_s[laneIdx] : 0.0f;
        sum = warpSum(sum);
        if (laneIdx == 0) atomicAdd(&sumSqGlobal, sum);
    }
}

__global__ void rmsNormalize(const float* input, float* output, int N, float gamma, float beta,
                             float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numVec = N / 4;
    int stride = blockDim.x * gridDim.x;
    
    float invRms = rsqrtf(sumSqGlobal / N + eps);

    for (int i = idx; i < numVec; i += stride) {
        float4 inVec = loadFloat4(input + 4 * i);
        float4 outVec = make_float4(inVec.x * invRms * gamma + beta, inVec.y * invRms * gamma + beta,
                                    inVec.z * invRms * gamma + beta, inVec.w * invRms * gamma + beta);
        storeFloat4(output + 4 * i, outVec);
    }
    for (int i = numVec * 4 + idx; i < N; i += stride) {
        output[i] = input[i] * invRms * gamma + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float zero = 0.0f;
    cudaMemcpyToSymbol(sumSqGlobal, &zero, sizeof(float));

    rmsReduction<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, gamma, beta, output, N, eps);
    rmsNormalize<<<blocksPerGrid, THREADS_PER_BLOCK>>>(input, output, N, gamma, beta, eps);
}
