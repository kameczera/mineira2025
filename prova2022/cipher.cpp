#include <vector>
#include <iostream>
#include <string>
using namespace std;

int main() {
    string crypted;
    int trilha;
    
    cin >> crypted;
    cin >> trilha;

    vector<char> decrypted;
    int crypt_cont = 0;
    int decryp_cont;
    
    int lim = trilha;
    for(int i = 0; i < lim; i++) {
        decryp_cont = i;
        int pulo = (trilha - 1) * 2 < 1 ? (trilha - 1) * 2 : 1;
        trilha--;
        while(decryp_cont < crypted.size()) {
            decrypted[decryp_cont] = crypted[crypt_cont++];
            decryp_cont += pulo;
        }
    }

    for(int i = 0; i < crypted.size(); i++) cout << decrypted[i];
}