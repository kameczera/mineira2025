#include <iostream>

using namespace std;

#define LEN 256
#define BLOCK_WIDTH 16

int* initialize_vec(int len) {
    int* vec = (int*) malloc(sizeof(int) * len);
    for(int i = 0; i < len; i++) {
        vec[i] = 1;
    }

    return vec;
}

void print_vec(int* vec, int len) {
    for(int i = 0; i < len; i++) {
        cout << vec[i];
    }
}

__global__ void sum_vec(int* vec1, int* vec2, int* ans) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < LEN) ans[idx] = vec1[idx] + vec2[idx];
}

int main() {
    int* vec1 = initialize_vec(LEN);
    int* vec2 = initialize_vec(LEN);
    int* d_vec1, *d_vec2, *d_ans;
    
    cudaMalloc(&d_vec1, sizeof(int) * LEN);
    cudaMemcpy(d_vec1, vec1, sizeof(int) * LEN, cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_vec2, sizeof(int) * LEN);
    cudaMemcpy(d_vec2, vec2, sizeof(int) * LEN, cudaMemcpyHostToDevice);

    cudaMalloc(&d_ans, sizeof(int) * LEN);

    dim3 tpb(BLOCK_WIDTH, 1);
    dim3 bpg((LEN + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 1);

    sum_vec<<<bpg, tpb>>>(d_vec1, d_vec2, d_ans);

    int* ans = (int*) malloc(sizeof(int) * LEN);
    cudaMemcpy(ans, d_ans, sizeof(int) * LEN, cudaMemcpyDeviceToHost);

    print_vec(ans, LEN);
}