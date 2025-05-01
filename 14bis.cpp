#include <iostream>

int sol(int height, int width, int* matrix) {
    int max_len = 0;
    // verificação por linha
    for(int i = 0; i < height; i++) {
        int act_len = 1;
        int h = matrix[i * width];
        for(int j = 1; j < width; j++) {
            if(std::abs(h - matrix[i * width + j]) <= 1) act_len += 1;
            else {
                if(act_len > max_len) max_len = act_len;
                act_len = 1;
            }
            h = matrix[i * width + j];
        }
        if(act_len > max_len) max_len = act_len;
    }
    
    /* otimização: se a maior pista encontrada na verificação de linha-por-linha for maior que
    o tamanho de colunas, não precisa verificar colunas
    */
    if (max_len > height) return max_len;

    // verificação por coluna
    for(int j = 0; j < width; j++) {
        int act_len = 1;
        int h = matrix[j];
        for(int i = 1; i < height; i++) {
            if(std::abs(h - matrix[i * width + j]) <= 1) act_len += 1;
            else {
                if(act_len > max_len) max_len = act_len;
                act_len = 1;
            }
            h = matrix[i * width + j];
        }
        if(act_len > max_len) max_len = act_len;
    }

    return max_len;
}

int main() {
    int width, height;
    std::cin >> height;
    std::cin >> width;
    int* matrix = (int*) malloc(sizeof(int) * height * width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            std::cin >> matrix[i * width + j];
        }
    }

    std::cout << sol(height, width, matrix);
}