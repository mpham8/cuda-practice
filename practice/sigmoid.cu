#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidFcn(const float x){
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float4 loadFloat4(const float* ptr){
    return reinterpret_cast<const float4*>(ptr)[0];
}

__device__ __forceinline__ void storeFloat4(float* ptr, float4& v){
    reinterpret_cast<float4*>(ptr)[0] = v;
}


__global__ void sigmoidKernel(const float* input, float* output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int fourIdx = 4*idx;

    if (fourIdx + 3 < N){
        float4 v = loadFloat4(input + fourIdx);
        v.x = sigmoidFcn(v.x);
        v.y = sigmoidFcn(v.y);
        v.z = sigmoidFcn(v.z);
        v.w = sigmoidFcn(v.w);
        storeFloat4(output + fourIdx, v);
    } else {
        for (int i = 0; i < 4; i++){
            int j = fourIdx + i;
            if (j < N) output[j] = sigmoidFcn(input[j]);
        }
    }
}

extern "C" void solve(const float* input, float* output, int N){
    int threadsPerBlock = 256;
    int sigmoidsPerBlock = (threadsPerBlock + 3)/4;
    int blocksPerGrid = (N + sigmoidsPerBlock - 1)/sigmoidsPerBlock;

    sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}