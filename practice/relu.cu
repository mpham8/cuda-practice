#include <cuda_runtime.h>

__device__ __forceinline__ float reluFcn(const float x){
    return fmaxf(x, 0);
}

//recast float ptr as float4 struct ptr
__device__ __forceinline__ float4 loadFloat4(const float* ptr) {
    return reinterpret_cast<const float4*>(ptr)[0];
}

//recast float ptr as float4 struct ptr and store float4 struct there
__device__ __forceinline__ void storeFloat4(float* ptr, const float4& v) {
    reinterpret_cast<float4*>(ptr)[0] = v;
}

__global__ void reluKernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //each thread handles 4 threads unless at boundary
    int fourIdx = 4 * idx;
    if (fourIdx + 3 < N){
        float4 v = loadFloat4(input + fourIdx);
        v.x = reluFcn(v.x);
        v.y = reluFcn(v.y);
        v.z = reluFcn(v.z);
        v.w = reluFcn(v.w);
        storeFloat4(output + fourIdx, v);
    } else {
        for (int i = 0; i < 4; i++) {
            int j = fourIdx + i;
            if (j < N) {
                output[j] = reluFcn(input[j]);
            }
        }

    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int relusPerBlock = threadsPerBlock * 4;
    int blocksPerGrid = (N + relusPerBlock - 1) / relusPerBlock;

    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N); 
    cudaDeviceSynchronize();

}
