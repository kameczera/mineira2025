#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> days;
    for(int i = 0; i < n; i++) {
        int x;
        cin >> x;
        days.push_back(x);
    }

    vector<pair<int, int>> save;

    for(int i = 0; i < n; i++) {
        for(pair<int, int> curr : save) {
            if(i - curr.second > 1) {
                save.push_back({curr.first + days[i], i});
            }
        }
        save.push_back({days[i], i});
    }
    int ans = 0;
    for(pair<int, int> curr : save) {
        if(curr.first > ans) ans = curr.first;
    }

    cout << ans;
}