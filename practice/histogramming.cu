#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK 256


//load int4
__global__ __inline__ int4 loadInt4(const int* ptr){
    return reinterpret_cast<const int4*>(ptr)[0];
}

//reduction
__global__ int histogram(const int* input, int* histogram, int N, int numBins) {
    int tId = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tId;
    int warpId = tId \\ 32;
    
    //init, zero out 8 shared mems
    extern __shared__ int tileS[];
    
    //each thread loads int4 at a time
    loadsPerThread = numBins \\ 32;
    for (int i = 0; i<y; i+=x){
        int4 v = loadInt4();
        tileS[warpId * numBins + tId * i] = v.x;
        tileS[warpId * numBins + tId * i + 1] = v.y;
        tileS[warpId * numBins + tId * i + 2] = v.z;
        tileS[warpId * numBins + tId * i + 3] = v.w;
    }

    //8 copies in shared mem -> 1 copy per block

    //global atomic add
}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int localCopiesPerBlock = THREADS_PER_BLOCK / 32;
    size_t sharedMemBytes = num_bins * localCopiesPerBlock * sizeof(int);
    
    // readsPerThread = 
    // int blocksPerGrid = (N + THREADS_PER_BLOCK - 1

    histogram<<<blocks, THREADS_PER_BLOCK, sharedMemBytes>>>(const int* input, int* histogram, int N, int numBins);
}
