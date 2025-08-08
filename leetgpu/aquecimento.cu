#include <iostream>
#include <cstdlib>

using namespace std;

#define TILE_WIDTH 32

__global__ void matrix_mul(float* matrix_a, float* matrix_b, float* matrix_ans, int n) {
    __shared__ float tiles_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tiles_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0;
    for(int i = 0; i < (TILE_WIDTH + n - 1) / TILE_WIDTH; i++) {
        if(row < n && i * TILE_WIDTH + threadIdx.x < n) tiles_a[threadIdx.y][threadIdx.x] = matrix_a[row * n + TILE_WIDTH * i + threadIdx.x];
        else tiles_a[threadIdx.y][threadIdx.x] = 0.0;
        if(col < n && i * TILE_WIDTH + threadIdx.y < n) tiles_b[threadIdx.y][threadIdx.x] = matrix_b[n * (i * TILE_WIDTH + threadIdx.y) + col];
        else tiles_b[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();
    
        for(int j = 0; j < TILE_WIDTH; j++) {
            sum += tiles_a[threadIdx.y][j] * tiles_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    if(row < n && col < n) matrix_ans[row * n + col] = sum;
}

int main() {
    int n;
    cin >> n;
    size_t s = sizeof(float) * n * n;
    float* matrix_a = (float*) malloc(s);
    float* matrix_b = (float*) malloc(s);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cin >> matrix_a[i * n + j];
        }
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cin >> matrix_b[i * n + j];
        }
    }

    float* d_matrix_a, *d_matrix_b, *d_matrix_ans;
    cudaMalloc(&d_matrix_a, s);
    cudaMalloc(&d_matrix_b, s);
    cudaMalloc(&d_matrix_ans, s);

    cudaMemcpy(d_matrix_a, matrix_a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b, s, cudaMemcpyHostToDevice);

    dim3 tpb(TILE_WIDTH,TILE_WIDTH);
    dim3 bpg((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_mul<<<bpg, tpb>>>(d_matrix_a, d_matrix_b, d_matrix_ans, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    float* matrix_ans = (float*) malloc(sizeof(float) * n * n);
    cudaMemcpy(matrix_ans, d_matrix_ans, s, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout << matrix_ans[i * n + j] << " ";
        }
        cout << "\n";
    }

    cudaFree(d_matrix_a); cudaFree(d_matrix_b); cudaFree(d_matrix_ans);
    free(matrix_a); free(matrix_b); free(matrix_ans);
}