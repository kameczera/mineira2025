#ifndef HELPS_CUH
#define HELPS_CUH

#include <iostream>

using namespace std;

__device__ const int tile_width = 32;

int* initialize_matrix(int rows, int cols) {
    int* matrix = (int*) malloc(sizeof(int) * rows * cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i * cols + j] = 1;
        }
    }
    return matrix;
}

// Correção: rows antes de cols; indexação em row-major
void print_matrix(const int* matrix, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

#endif