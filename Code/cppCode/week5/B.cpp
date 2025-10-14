#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

const int MAX_N = 1005;
const int MAX_B = 10005;  // 增大背包容量上限以适应输入数据

int n, b;
int w[MAX_N];  // 物品重量
int v[MAX_N];  // 物品价值
int dp[MAX_B];  // 使用一维数组优化空间，dp[j]表示容量为j时的最大价值

int main() {
    cin >> n >> b;
    
    // 读取物品数据
    for (int i = 1; i <= n; i++) {
        cin >> w[i] >> v[i];
    }
    
    // 初始化dp数组为0
    memset(dp, 0, sizeof(dp));
    
    // 0-1背包核心算法，使用一维数组优化
    for (int i = 1; i <= n; i++) {
        //从大到小遍历，防止物品被重复使用
        for (int j = b; j >= w[i]; j--) {
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    
    cout << dp[b] << endl;
    return 0;
}
