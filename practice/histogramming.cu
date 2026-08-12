#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256


//reduction
__global__ void histogramKernel(const int* __restrict__ input, int* histogram, int N, int numBins) {
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
    int n4 = N / 4;
    const int4* input4 = reinterpret_cast<const int4*>(input);
    for (int i = idx; i < n4; i += blockDim.x * gridDim.x) {
        int4 v = __ldg(input4 + i);
        atomicAdd(&sharedMem[warpId * numBins + v.x], 1);
        atomicAdd(&sharedMem[warpId * numBins + v.y], 1);
        atomicAdd(&sharedMem[warpId * numBins + v.z], 1);
        atomicAdd(&sharedMem[warpId * numBins + v.w], 1);
    }
    for (int i = n4 * 4 + idx; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(&sharedMem[warpId * numBins + input[i]], 1);
    }
    __syncthreads();

    //8 copies in shared mem -> 1 copy per block
    for (int i = tId; i < numBins; i += blockDim.x){
        #pragma unroll
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

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int blocksPerGrid = min(numSMs * 4, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    histogramKernel<<<blocksPerGrid, THREADS_PER_BLOCK, sharedMemBytes>>>(input, histogram, N, num_bins);
}
