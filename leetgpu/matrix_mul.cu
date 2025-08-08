#include <iostream>
#include <cstdlib>
#include "helps.cuh"
#define TILE_WIDTH 32
static const int M = 35;  // M
static const int K = 35;   // K
static const int N = 35;   // N


__global__ void fused_matmul_transpose_softmax(float* q, float* k, float* ans, int m, int n) {
    __shared__ float tile_q[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_k[TILE_WIDTH][TILE_WIDTH];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0;
    for(int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        if(row < m && TILE_WIDTH * i + threadIdx.x < n) tile_q[threadIdx.y][threadIdx.x] = q[row * n + TILE_WIDTH * i + threadIdx.x];
        else tile_q[threadIdx.y][threadIdx.x] = 0.0;
        if(col < m && TILE_WIDTH * i + threadIdx.y < n) tile_k[threadIdx.y][threadIdx.x] = k[col * n + TILE_WIDTH * i + threadIdx.y];
        else tile_k[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();
        
        for(int j = 0; j < TILE_WIDTH; j++) {
            sum += tile_q[threadIdx.y][j] * tile_k[j][threadIdx.x];
        }
        __syncthreads();
    }

}

int main() {
    int* A = initialize_matrix(M, K);
    int* B = initialize_matrix(K, N);
    int* C = (int*)malloc(sizeof(int) * M * N);

    int* d_A; int* d_B; int* d_C;
    cudaMalloc(&d_A, sizeof(int) * M * K);
    cudaMalloc(&d_B, sizeof(int) * K * N);
    cudaMalloc(&d_C, sizeof(int) * M * N);

    cudaMemcpy(d_A, A, sizeof(int) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * K * N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(tile_width, tile_width);
    dim3 blocksPerGrid((N + tile_width - 1) / tile_width, (M + tile_width - 1) / tile_width);

    fused_matmul_transpose_softmax<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(C, d_C, sizeof(int) * M * N, cudaMemcpyDeviceToHost);

    // Print resultado
    std::cout << "Resultado:\n";
    print_matrix(C, M, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}
