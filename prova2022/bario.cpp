#include <vector>
#include <iostream>

using namespace std;

int main() {
    int n;
    vector<char> s;
    cin >> n;
    for(int i = 0; i < n; i++) {
        char b;
        cin >> b;
        s.push_back(b);
    }
    int i = 0;
    int v = 0;
    int n_jumps = 0;
    while(i < s.size()) {
        if(s[i] == 'x') {
            v++;
            i++;
        }
        else {
            n_jumps++;
            int jump = 0;
            while(i < s.size() && s[i] == '.') {
                jump++;
                i++;
            }
            if(jump > v) {
                cout << "-1";
                return 0;
            }
            v = 0;
        }
    }
    cout << n_jumps;
    return 0;
}