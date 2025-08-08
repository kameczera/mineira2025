#include <iostream>
#include <cstdlib>
#include "helps.cuh"

#define TILE_DIM 16
#define ROWS 2
#define COLS 3

__global__ void transpose_matrix(const int* A, int* B, int M, int N) {
    __shared__ int tile[TILE_DIM][TILE_DIM + 1];
    int row = blockDim.y * TILE_DIM + threadIdx.y;
    int col = blockDim.x * TILE_DIM + threadIdx.x;

    if(col < N && row < M) tile[threadIdx.y][threadIdx.x] = A[y * N + x];

    __syncthreads();

    row = blockIdx.y * 
}

int main() {
    int* matrix = initialize_matrix(ROWS, COLS);
    int* d_matrix;
    int* d_transposed;
    
    cudaMalloc(&d_matrix, sizeof(int) * ROWS * COLS);
    cudaMalloc(&d_transposed, sizeof(int) * ROWS * COLS);
    cudaMemcpy(d_matrix, matrix, sizeof(int) * ROWS * COLS, cudaMemcpyHostToDevice);
    
    dim3 tpb(16, 16);
    dim3 bpg((COLS + tpb.x - 1) / tpb.x, (ROWS + tpb.y - 1) / tpb.y);

    transpose_matrix<<<bpg, tpb>>>(d_matrix, d_transposed, ROWS, COLS);
    cudaDeviceSynchronize();

    int *transposed = (int*) malloc(sizeof(int) * ROWS * COLS);
    cudaMemcpy(transposed, d_transposed, sizeof(int) * ROWS * COLS, cudaMemcpyDeviceToHost);

    std::cout << "Matriz original:\n";
    print_matrix(matrix, ROWS, COLS);
    std::cout << "\nMatriz transposta:\n";
    print_matrix(transposed, COLS, ROWS);

    cudaFree(d_matrix);
    cudaFree(d_transposed);
    free(matrix);
    free(transposed);

    return 0;
}
