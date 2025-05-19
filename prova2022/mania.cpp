#include <iostream>

using namespace std;

int main() {
    int g;
    cin >> g;
    int cont = 0;
    char t;
    while(cin >> t) {
        if(t == 'D') cont += g;
        else if(t == 'E') cont -= g;

        if(cont >= 360 || cont <= -360) {
            cout << "S";
            return 0;
        }
    }

    cout << "N";
    return 0;
}