#include <cuda_runtime.h>

#define STRIDE 4
#define THREADS_PER_BLOCK 256
#define SHARED_MEM_SIZE (THREADS_PER_BLOCK/32)


__device__ __forceinline__ float4 loadFloat4(const float* ptr){
    //recast float ptr as float4 ptr
    return reinterpret_cast<const float4*>(ptr)[0];
}

__device__ __forceinline__ float warpReduction(float val){
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void dotProduct(const float* A, const float* B, float* result, const int N){

    int tId = threadIdx.x;
    int warpIdx = tId / 32;
    int laneIdx = tId % 32;
    int idx = blockIdx.x * blockDim.x + tId;

    __shared__ float data_s[SHARED_MEM_SIZE];

    //global stride 4 dot products
    float sum = 0.0f;
    for (int i = idx * 4; i < N; i += blockDim.x * gridDim.x * 4){
        if (i + 3 < N){
            //convert to float 4 pointer
            float4 a = loadFloat4(A + i);

            float4 b = loadFloat4(B + i);

            //dot the float4
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

        } else {
            for (int j = i; j < N; j++) sum += A[j] * B[j];
        }
    }

    //reduction

    //warp reduction 256 -> 8
    sum = warpReduction(sum);

    //put 8 into shared mem
    if (laneIdx == 0) data_s[warpIdx] = sum;
    __syncthreads();

    //warp reduction 8->1
    sum = (laneIdx < SHARED_MEM_SIZE) ? data_s[laneIdx] : 0.0f;
    if (warpIdx == 0){
        sum = warpReduction(sum);
        if (laneIdx == 0) atomicAdd(result, sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int loadsPerBlock = THREADS_PER_BLOCK * STRIDE * 4;
    int blocksPerGrid = (N + loadsPerBlock - 1)/loadsPerBlock;

    dotProduct<<<blocksPerGrid, THREADS_PER_BLOCK>>>(A, B, result, N);
}
