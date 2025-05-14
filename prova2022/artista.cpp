#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> k(n);
    
    for(int i = 0; i < n; i++) {
        cin >> k[i];
    }
    vector<pair<int, int>> sums;
    for(int i = 0; i < n; i++) {
        for(pair<int, int> sum : sums) {
            if(i - sum.second > 1) {
                pair<int, int> new_sum = {sum.first + k[i], i};
                sums.push_back(new_sum);
            }
        }
        pair<int, int> sum = {k[i], i};
        sums.push_back(sum);
    }
    int highest = 0;
    for(pair<int, int> p : sums){
        if(p.first > highest) highest = p.first;
    }

    cout << highest;
    return 0;
}