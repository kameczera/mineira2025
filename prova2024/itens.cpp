#include <iostream>
#include <vector>
#include <map>

using namespace std;

int main() {
    int width, height, time, pos_x, pos_y;
    cin >> width;
    cin >> height;
    cin >> time;
    cin >> pos_x;
    cin >> pos_y;

    int** grid = (int**) malloc(sizeof(int*) * width);
    for(int i = 0; i < width; i++) {
        grid[i] = (int*) malloc(sizeof(int) * height);
    }
    int good_squares = 0;
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            char input;
            cin >> input;
            if(input == '#') grid[i][j] = 1;
            else {
                grid[i][j] = 0;
                good_squares += 1;
            }
        }
    }
    vector<tuple<int, int>> stack_squares;
    stack_squares.push_back(make_tuple(pos_y * height + pos_x, 0));
    map<int, bool> visited_squares;
    int distance = 0;
    while(stack_squares.empty()) {
        bool has_visited = false;
        tuple<int, int> curr_square = stack_squares.back();
        int id = get<0>(curr_square);
        stack_squares.pop_back();
        if(visited_squares[id]) has_visited = true; 
        if(!has_visited) {
            int x = id % height;
            int y = id / width;
            visited_squares[id] = true;
            if(x < width) stack_squares.push_back(make_tuple(x + 1, y));
            if(x > 0) stack_squares.push_back(make_tuple(x + 1, y));
            if(x < width) stack_squares.push_back(make_tuple(x + 1, y));
        }
    } 
}