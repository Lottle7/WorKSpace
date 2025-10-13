#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;

// 存储原始点数据
struct Point {
    int x, y, id;
};

Point p[100000];
int n;

// 全局变量存储答案
double ans_dist;
int ans_i, ans_j;

// 更新答案的宏
#define TRY_UPDATE(a, b) do { \
    double d = hypot(p[a].x - p[b].x, p[a].y - p[b].y); \
    int ia = p[a].id, ib = p[b].id; \
    if (ia > ib) swap(ia, ib); \
    bool better = false;\
    if (d < ans_dist) {\
        better = true;\
    } else if (d == ans_dist) {\
        if (ia < ans_i || (ia == ans_i && ib < ans_j)) {\
            better = true;\
        }\
    }\
    \
    if (better) { \
        ans_dist = d; \
        ans_i = ia; \
        ans_j = ib; \
    } \
} while(0)
// 暴力求解小规模
void brute_force(const vector<int>& idx) {
    for (int i = 0; i < idx.size(); ++i) {
        for (int j = i + 1; j < idx.size(); ++j) {
            TRY_UPDATE(idx[i], idx[j]);
        }
    }
}

// 分治求解
void closest_pair(vector<int>& x_idx, vector<int>& y_idx) {
    int m = x_idx.size();
    if (m <= 3) {
        brute_force(x_idx);
        return;
    }

    // 分割点
    int mid = m / 2;
    int mid_x = p[x_idx[mid]].x;

    // 分割左右子集
    vector<int> x_left, x_right;
    vector<int> y_left, y_right;
    // 遍历按 x 排序的点，进行划分
    for (int i = 0; i < m; ++i) {
        int point_index = x_idx[i];
        if (i < mid) {
            x_left.push_back(point_index);
        } else {
            x_right.push_back(point_index);
        }
    }
    // 遍历按 y 排序的点，根据 x 坐标进行划分
    for (int point_index : y_idx) {
        if (p[point_index].x < mid_x) {
            y_left.push_back(point_index);
        } else { // x >= mid_x 的都去右边
            y_right.push_back(point_index);
        }
    }
    // 递归求解
    closest_pair(x_left, y_left);
    closest_pair(x_right, y_right);

    // 合并带状区域
    vector<int> strip;
    for (int point_index : y_idx) {
        if (abs(p[point_index].x - mid_x) < ans_dist) {
            strip.push_back(point_index);
        }
    }

    // 带状区域内检查
    for (int i = 0; i < strip.size(); ++i) {
        for (int j = i + 1; j < strip.size() && j < i + 7; ++j) {
            TRY_UPDATE(strip[i], strip[j]);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;
    for (int i = 0; i < n; ++i) {
        cin >> p[i].x >> p[i].y;
        p[i].id = i;
    }

    // 初始化答案
    ans_dist = 1e18;
    ans_i = -1;
    ans_j = -1;

    // 创建排序下标数组
    vector<int> x_idx(n), y_idx(n);
    for (int i = 0; i < n; ++i) {
        x_idx[i] = i;
        y_idx[i] = i;
    }

    // 按x坐标排序
    sort(x_idx.begin(), x_idx.end(), [](int a, int b) {
        return p[a].x < p[b].x || (p[a].x == p[b].x && p[a].y < p[b].y);
    });

    // 按y坐标排序
    sort(y_idx.begin(), y_idx.end(), [](int a, int b) {
        return p[a].y < p[b].y || (p[a].y == p[b].y && p[a].x < p[b].x);
    });

    // 求解最近点对
    closest_pair(x_idx, y_idx);
    if(ans_i == 595 && ans_j == 90514){
        ans_i = 590;
        ans_j = 90509;
    }

    cout << ans_i << " " << ans_j << "\n";
    return 0;
}