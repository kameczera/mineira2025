#include <iostream>

void sol(int n, int sum) {
    for(int i = 1; i < n; i++) {
        int fingers;
        std::cin >> fingers;
        sum += fingers;
    }
    for(int i = 0; i < n; i++) {
        if(sum % n <= 20) std::cout << sum % n;
        else std::cout << -1;
        std::cout << "\n";
        sum++;
    }
}

int main() {
    int n;
    std::cin >> n;
    int sum = 0;
    sol(n, sum);
    return 0;
}