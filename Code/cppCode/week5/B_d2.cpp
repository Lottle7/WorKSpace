#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

const int MAX_N = 1005;
const int MAX_B = 10005;  // 确保容量上限足够大

int n, b;
int w[MAX_N];  // 物品重量
int v[MAX_N];  // 物品价值
int dp[MAX_N][MAX_B];  // dp[i][j]表示前i个物品在容量j下的最大价值

int main() {
    cin >> n >> b;
    
    // 读取物品数据
    for (int i = 1; i <= n; i++) {
        cin >> w[i] >> v[i];
    }
    
    // 初始化：第0行（没有物品）和第0列（容量为0）都为0
    memset(dp, 0, sizeof(dp));
    
    // 二维数组实现0-1背包
    for (int i = 1; i <= n; i++) {
        // 遍历所有可能的容量
        for (int j = 1; j <= b; j++) {
            // 如果当前物品重量超过背包容量，无法放入
            if (w[i] > j) {
                dp[i][j] = dp[i-1][j];
            } else {
                // 否则选择放入或不放入的最大值
                dp[i][j] = max(dp[i-1][j], dp[i-1][j - w[i]] + v[i]);
            }
        }
    }
    
    cout << dp[n][b] << endl;
    return 0;
}
