#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

int gcd(int a, int b) {
    while (b) {
        int tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

int main() {
    int N, M, T, ci, cj;
    cin >> N >> M >> T >> ci >> cj;
    ci--; cj--; // Convertendo para 0-based

    vector<vector<char>> grid(N, vector<char>(M));
    int total_empty = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            cin >> grid[i][j];
            if (grid[i][j] == '.') total_empty++;
        }
    }

    vector<vector<int>> visited(N, vector<int>(M, -1));
    queue<pair<int, int>> q;

    visited[ci][cj] = 0;
    q.push({ci, cj});
    int reachable = 0;

    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    while (!q.empty()) {
        pair<int, int> front = q.front(); q.pop();
        int x = front.first;
        int y = front.second;
        int dist = visited[x][y];
        if (dist > T) continue;

        if (grid[x][y] == '.') reachable++;

        for (int dir = 0; dir < 4; ++dir) {
            int nx = x + dx[dir];
            int ny = y + dy[dir];
            if (nx >= 0 && nx < N && ny >= 0 && ny < M && grid[nx][ny] == '.' && visited[nx][ny] == -1) {
                visited[nx][ny] = dist + 1;
                q.push({nx, ny});
            }
        }
    }

    if (reachable == 0) {
        cout << "0 1\n";
    } else {
        int divisor = gcd(reachable, total_empty);
        cout << (reachable / divisor) << " " << (total_empty / divisor) << "\n";
    }

    return 0;
}
