#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

using namespace std;

#define TILE_WIDTH 32

__global__ void softmax(float* matrix, int m, int n) {
    float max_value;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < n && col < m) {
        if(col == 0) {
            max_value = matrix[row * n];
            for(int i = 1; i < m; i++) {
                if(matrix[row * n + i] > max_value) max_value = matrix[row * n + i];
            }
        }
        __syncthreads();
        float sum_exp = 0.0f;
        for (int i = 0; i < n; i++) {
            float val = matrix[row * n + i];
            sum_exp += expf(val - max_value);
        }

        for (int i = 0; i < n; i++) {
            float val = matrix[row * n + i];
            matrix[row * n + i] = expf(val - max_value) / sum_exp;
        }
    }
}

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
    if (row < m && col < m) ans[row * m + col] = sum / sqrtf(n);
    
    float max_value;
    if(row < m && col < m) {
        if(col == 0) {
            max_value = ans[row * m];
            for(int i = 1; i < m; i++) {
                if(ans[row * m + i] > max_value) max_value = ans[row * m + i];
            }
        }
        __syncthreads();
        float sum_exp = 0.0f;
        for (int i = 0; i < m; i++) {
            float val = ans[row * m + i];
            sum_exp += expf(val - max_value);
        }

        for (int i = 0; i < m; i++) {
            float val = ans[row * m + i];
            ans[row * m + i] = expf(val - max_value) / sum_exp;
        }
    }
}

__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float tiles_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tiles_b[TILE_WIDTH][TILE_WIDTH];
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0;
    for(int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        if(row < M && i * TILE_WIDTH + threadIdx.x < K) tiles_a[threadIdx.y][threadIdx.x] = A[row * K + TILE_WIDTH * i + threadIdx.x];
        else tiles_a[threadIdx.y][threadIdx.x] = 0.0;
        if(i * TILE_WIDTH + threadIdx.y < K && col < N) tiles_b[threadIdx.y][threadIdx.x] = B[N * (threadIdx.y + TILE_WIDTH * i) + col];
        else tiles_b[threadIdx.y][threadIdx.x] = 0.0;
        
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; j++) {
            sum += tiles_a[threadIdx.y][j] * tiles_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < M && col < N)
        C[row*N + col] = sum;
}

void printMatrix(float* mat, int rows, int cols) {
    cout << fixed << setprecision(2);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cout << mat[i * cols + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int m, n;
    cin >> m >> n;

    size_t matrix_in = sizeof(float) * m * n;

    float* h_q = (float*) malloc(matrix_in);
    float* h_k = (float*) malloc(matrix_in);
    float* h_v = (float*) malloc(matrix_in);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            cin >> h_q[i*n + j];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            cin >> h_k[i*n + j];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            cin >> h_v[i*n + j];

    float* d_q, *d_k, *d_v, *d_softmax_ans, *d_ans;
    cudaMalloc(&d_q, matrix_in);
    cudaMalloc(&d_k, matrix_in);
    cudaMalloc(&d_v, matrix_in);
    cudaMalloc(&d_softmax_ans, sizeof(float) * m * m);
    cudaMalloc(&d_ans, sizeof(float) * m * n);

    cudaMemcpy(d_q, h_q, matrix_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, matrix_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, matrix_in, cudaMemcpyHostToDevice);

    dim3 tpb(TILE_WIDTH, TILE_WIDTH);
    dim3 bpg((m + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    fused_matmul_transpose_softmax<<<bpg, tpb>>>(d_q, d_k, d_softmax_ans, m, n);
    cudaDeviceSynchronize();
    
    // softmax<<<bpg, tpb>>>(d_softmax_ans, m, m);
    // cudaDeviceSynchronize();
    
    dim3 bpg_2((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul<<<bpg_2, tpb>>>(d_softmax_ans, d_v, d_ans, m, m, n);
    cudaDeviceSynchronize();

    float* h_ans = (float*) malloc(sizeof(float) * m * n);
    cudaMemcpy(h_ans, d_ans, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    
    cout << "Resultado da matriz (q * k^T):" << endl;
    printMatrix(h_ans, m, n);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_ans);

    return 0;
}