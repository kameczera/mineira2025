#include <iostream>

typedef struct {
    int is_on;
    node* sons;
} node;


int main() {
    node* root;
    int n;
    std::cin >> n;
    node** p = (node**) malloc(sizeof(node*) * n);
    for(int i = 0; i < n; i++) {
        p[i] = (node*) malloc(sizeof(node));
    }
    


    return 0;
}