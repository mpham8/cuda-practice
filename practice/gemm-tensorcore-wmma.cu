#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define THREAD_PER_BLOCK 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define C_DIM_PER_BLOCK 128
#define WARP_COL 64
#define WARP_ROW 32
#define WARP_C_FRAGS ((WARP_ROW / WMMA_M) * (WARP_COL / WMMA_N))

__global__ void gemmTensorcoreWmma(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // calc ids
    int tIdx = threadIdx.x;
    int warpIdx = tIdx / 32;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int blockCRow = blockRow * C_DIM_PER_BLOCK;
    int blockCCol = blockCol * C_DIM_PER_BLOCK;

    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;

    int warpCRow = blockCRow + warpRow * WARP_ROW;
    int warpCCol = blockCCol + warpCol * WARP_COL;


    //init shared mem to store part of A, B
    __shared__ half As[C_DIM_PER_BLOCK][WMMA_K];
    __shared__ half Bs[C_DIM_PER_BLOCK][WMMA_K]; 


    //init frags and set to 0
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrags[WARP_C_FRAGS];

    #pragma unroll
    for (int i = 0; i < WARP_C_FRAGS; i++) {
        wmma::fill_fragment(cFrags[i], 0.0f);
    }

    //matmul A*B
    for (int kTile = 0; kTile < K; kTile += WMMA_K) {
        //load part A into shared mem
        for (int i = tIdx; i < C_DIM_PER_BLOCK * WMMA_K; i += blockDim.x) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int gr = blockCRow + row;
            int gk = kTile + col;
            As[row][col] = (gr < M && gk < K) ? A[gr * K + gk] : __float2half(0.0f);
        }
        //load part B into shared mem
        for (int i = tIdx; i < WMMA_K * C_DIM_PER_BLOCK; i += blockDim.x) {
            int kLocal = i / C_DIM_PER_BLOCK;
            int col = i % C_DIM_PER_BLOCK;
            int gk = kTile + kLocal;
            int gc = blockCCol + col;
            Bs[col][kLocal] = (gk < K && gc < N) ? B[gk * N + gc] : __float2half(0.0f);
        }

        __syncthreads();

        //fragment matmul
        #pragma unroll
        for (int i = 0; i < WARP_C_FRAGS; i++) {
            int tileRow = i / (WARP_COL / WMMA_N);
            int tileCol = i % (WARP_COL / WMMA_N);
            int aRow = warpRow * WARP_ROW + tileRow * WMMA_M;
            int bCol = warpCol * WARP_COL + tileCol * WMMA_N;

            wmma::load_matrix_sync(aFrag, &As[aRow][0], WMMA_K);
            wmma::load_matrix_sync(bFrag, &Bs[bCol][0], WMMA_K);
            wmma::mma_sync(cFrags[i], aFrag, bFrag, cFrags[i]);
        }
    }

    __shared__ float tempC[THREAD_PER_BLOCK / 32][WMMA_M][WMMA_N];

    #pragma unroll
    for (int i = 0; i < WARP_C_FRAGS; i++) {
        int tileRow = i / (WARP_COL / WMMA_N);
        int tileCol = i % (WARP_COL / WMMA_N);
        int gRow = warpCRow + tileRow * WMMA_M;
        int gCol = warpCCol + tileCol * WMMA_N;

        wmma::store_matrix_sync(&tempC[warpIdx][0][0], cFrags[i], WMMA_N, wmma::mem_row_major);

        for (int j = tIdx % 32; j < WMMA_M * WMMA_N; j += 32) {
            int r = j / WMMA_N;
            int c = j % WMMA_N;
            int gr = gRow + r;
            int gc = gCol + c;
            if (gr < M && gc < N) {
                float prev = (beta != 0.0f) ? __half2float(C[gr * N + gc]) : 0.0f;
                C[gr * N + gc] = __float2half(alpha * tempC[warpIdx][r][c] + beta * prev);
            }
        }
        __syncwarp();
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 block(THREAD_PER_BLOCK);
    dim3 grid((N + C_DIM_PER_BLOCK - 1) / C_DIM_PER_BLOCK,
              (M + C_DIM_PER_BLOCK - 1) / C_DIM_PER_BLOCK);
    gemmTensorcoreWmma<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
