
#include <cuda_runtime.h>

//gated error gated linear unit (geglu)

__device__ loadFloat4X1(){

}

__device__ loadFloat4X2(){

}

__device__ storeFloat4(){

}

__device__ __forceinline__ float geluFcn(const float x){
    return 0.5f * x * (1.0f + erff(x/sqrtf(2.0f)));
}

__global__ void solve(const float* input, float* output, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx * 8 + 7 < N){
        float4 x1 = loadFloat4X1(input + 4*idx);
        float4 x2 = loadFloat4X2(input + N/2 + 4*idx);

        x1.x *= geluFcn(x2.x);
        x1.y *= geluFcn(x2.y);
        x1.z *= geluFcn(x2.z);
        x1.w *= geluFcn(x2.w);

        storeFloat4(output + 4*idx, x1);
        
    } else {
        for (int i = 0; i < 4; i++){
            int j = idx*4 + i;
            if (j < N/2) {
                output[j] = input[j] * geluFcn(input[N/2 + j]);
            }
        }
    }
}


extern "C" void solve(const float* input, float* output, int N){
    int threadsPerBlock = 256;
    int indicesPerBlock = threadsPerBlock * 8;
    int blocksPerGrid = (N  + indicesPerBlock - 1) / indicesPerBlock;

    solve<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}