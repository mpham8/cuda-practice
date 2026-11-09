#include <cuda_runtime.h>


__device__ __forceinline__ float4 loadFloat4(const float* ptr){
    return reinterpret_cast<const float4*>(ptr)[0];
}


__device__ __forceinline__ void storeFloat4(float* ptr, const float4& v){
    reinterpret_cast<float4*>(ptr)[0] = v;
}


__device__ __forceinline__ float sigmoidFcn(float x) {
    return 1.0f / (1.0f + __expf(-x));
}


__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    //load respective elements as float4 from x1 and x2 if 16 byte aligned address
    if (halfN % 4 == 0 && idx * 4 + 3 < halfN){
        float4 x1 = loadFloat4(input + idx * 4);
        float4 x2 = loadFloat4(input + halfN + idx * 4);
    
        float4 v;
    
        v.x = x1.x * sigmoidFcn(x1.x) * x2.x;
        v.y = x1.y * sigmoidFcn(x1.y) * x2.y;
        v.z = x1.z * sigmoidFcn(x1.z) * x2.z;
        v.w = x1.w * sigmoidFcn(x1.w) * x2.w;

        //store to global
        storeFloat4(output + idx * 4, v);
    } else {
        for (int k = 0; k < 4; k++){
            int i = idx * 4 + k;
            if (i >= halfN) break;

            float x1 = input[i];
            float x2 = input[halfN + i];
            
            //store to global
            output[i] = x1 * sigmoidFcn(x1) * x2;
        }
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int float4Chunks = (halfN + 3) / 4;
    int blocksPerGrid = (float4Chunks + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
