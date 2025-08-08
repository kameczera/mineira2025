#include <vector>
#include <iostream>

using namespace std;

int main() {
    int n, m, k, q;
    cin >> n, m, k, q;
    vector<vector<int>> grid(n, vector<int>(m));
    vector<vector<int>> data(q, vector<int>(3));
    int a, b, d;
    for(int i = 0; i < q; i++) {
        cin >> a >> b >> d;
        data[i] = {a, b, d};
        
    }
}