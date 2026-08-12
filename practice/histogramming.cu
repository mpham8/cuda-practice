#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256


//load int4
__device__ __forceinline__ int4 loadInt4(const int* ptr){
    return reinterpret_cast<const int4*>(ptr)[0];
}

//reduction
__global__ void histogramKernel(const int* input, int* histogram, int N, int numBins) {
    int tId = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tId;
    int warpId = tId / 32;
    
    //init, zero out 8 local copies in shared mem
    int warps = THREADS_PER_BLOCK / 32;
    int sharedMemSize = warps * numBins;
    extern __shared__ int sharedMem[];
    for (int i=tId; i<sharedMemSize; i+=THREADS_PER_BLOCK) sharedMem[i] = 0;
    __syncthreads();

    //each thread loads int4 at a time
    for (int i = idx * 4; i < N; i += blockDim.x * gridDim.x * 4) {
        if (i + 3 < N) {
            int4 v = loadInt4(input + i);
            atomicAdd(&sharedMem[warpId * numBins + v.x], 1);
            atomicAdd(&sharedMem[warpId * numBins + v.y], 1);
            atomicAdd(&sharedMem[warpId * numBins + v.z], 1);
            atomicAdd(&sharedMem[warpId * numBins + v.w], 1);
        } else {
            for (int j = i; j < N; ++j) {
                atomicAdd(&sharedMem[warpId * numBins + input[j]], 1);
            }
        }
    }
    __syncthreads();

    //8 copies in shared mem -> 1 copy per block
    for (int i = tId; i < numBins; i += blockDim.x){
        int sum = 0;
        for (int j = 0; j < warps; j++){
            sum += sharedMem[j * numBins + i];
        }
        //global atomic add
        atomicAdd(&histogram[i], sum);
    }
    
}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int localCopiesPerBlock = THREADS_PER_BLOCK / 32;
    size_t sharedMemBytes = num_bins * localCopiesPerBlock * sizeof(int);

    int elementsPerBlock = THREADS_PER_BLOCK * 4;
    int blocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;

    histogramKernel<<<blocksPerGrid, THREADS_PER_BLOCK, sharedMemBytes>>>(input, histogram, N, num_bins);
}
