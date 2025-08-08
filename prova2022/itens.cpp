#include <iostream>
#include <vector>
#include <queue>
#include <map>

using namespace std;

int gcd(int a, int b) {
    while(b) {
        int tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

int main() {
    int width, height, time, x, y;
    cin >> width >> height >> time >> x >> y;
    int total_empty = 0;
    vector<vector<char>> grid(width, vector<char>(height));
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            cin >> grid[i][j];
            if(grid[i][j] == '.') total_empty++;
        }
    }
    queue<pair<int, int>> q;
    vector<vector<int>> visited(width,vector<int>(height));
    q.push(make_pair(x, y));
    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};
    int reachable = 0;
    while(!q.empty()) {
        pair<int, int> front = q.front(); q.pop();
        int x = front.first;
        int y = front.second;
        int dist = visited[x][y];
        if(dist > time) continue;

        if(grid[x][y] == '.') reachable++;

        for(int dir = 0; dir < 4; ++dir) {
            int nx = x + dx[dir];
            int ny = y + dy[dir];
            if(nx >= 0 && nx < width && ny < height && grid[nx][ny] == '.' && visited[nx][ny] == -1) {
                visited[nx][ny] = dist + 1;
                q.push({nx, ny});
            }
        }
    }

    if(reachable == 0) cout << "0 1\n";
    else {
        int divisor = gcd(reachable, total_empty);
        cout << (reachable / divisor) << " " << (total_empty / divisor) << "\n";
    }

    return 0;
}   