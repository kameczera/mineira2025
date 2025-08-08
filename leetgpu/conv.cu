#include <iostream>
#include "helps.cuh"

#define N 35
#define N_MASK 3
#define RADIUS (N_MASK / 2)
#define TILE_WIDTH 32
#define BLOCK_WIDTH (TILE_WIDTH + 2 * RADIUS)

using namespace std;

__global__ void f_conv_matrix(int* matrix, int* conv_matrix, int* d_mask) {
    __shared__ int tile_matrix[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    int row_i = row_o - RADIUS;
    int col_i = col_o - RADIUS;

    // Carrega dados da matriz global para a memória compartilhada
    if (row_i >= 0 && row_i < N && col_i >= 0 && col_i < N)
        tile_matrix[ty][tx] = matrix[row_i * N + col_i];
    else
        tile_matrix[ty][tx] = 0;

    __syncthreads();

    int sum = 0;
    if (ty >= RADIUS && ty < (TILE_WIDTH + RADIUS) &&
        tx >= RADIUS && tx < (TILE_WIDTH + RADIUS) &&
        row_o < N && col_o < N) {
        for (int i = -RADIUS; i <= RADIUS; i++) {
            for (int j = -RADIUS; j <= RADIUS; j++) {
                sum += tile_matrix[ty + i][tx + j] *
                       d_mask[(i + RADIUS) * N_MASK + (j + RADIUS)];
            }
        }
        conv_matrix[row_o * N + col_o] = sum;
    }
}

int* initialize_mask(int n) {
    int* mask = (int*) malloc(sizeof(int) * n * n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            mask[i * n + j] = 1;
        }
    }
    return mask;
}

int main() {
    int* matrix = initialize_matrix(N, N);
    int* conv_matrix = initialize_matrix(N, N);
    int* mask = initialize_mask(N_MASK);

    int *d_matrix, *d_conv_matrix, *d_mask;
    cudaMalloc(&d_matrix, sizeof(int) * N * N);
    cudaMalloc(&d_conv_matrix, sizeof(int) * N * N);
    cudaMalloc(&d_mask, sizeof(int) * N_MASK * N_MASK);

    cudaMemcpy(d_matrix, matrix, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeof(int) * N_MASK * N_MASK, cudaMemcpyHostToDevice);

    dim3 tpb(BLOCK_WIDTH, BLOCK_WIDTH); // Total de threads por bloco (incluindo bordas)
    dim3 bpg((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    f_conv_matrix<<<bpg, tpb>>>(d_matrix, d_conv_matrix, d_mask);
    cudaDeviceSynchronize(); // Garante que o kernel terminou

    cudaMemcpy(conv_matrix, d_conv_matrix, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

    print_matrix(conv_matrix, N, N);

    cudaFree(d_matrix);
    cudaFree(d_conv_matrix);
    cudaFree(d_mask);
    free(matrix);
    free(conv_matrix);
    free(mask);

    return 0;
}
