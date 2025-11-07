// 最长上升子序列 需要nlogn的解法
#include <iostream>
#include <algorithm>
#include <cstring>
#include <math.h>
using namespace std;

int n;
int mod = 1e9 + 7;
long long a[1001];
int b;
int ans = 0;
int dp[1001];
int main(){
    cin >> n >> b;
    a[1] = b;
    
    // 1-n 下标从1开始
    for(int i = 2; i <= n; i ++) {
        a[i] = 1LL * (a[i - 1] + 1) * (a[i - 1] + 1) % mod;
    }
    for(int k = 1;k <= n;k++){
        dp[k] = 1;//以自己结尾 至少有一的长度
        for(int j = 1;j < k; j++){
            if(a[j] < a[k] && dp[j] + 1 > dp[k]){//a[j] < a[k] 递增 dp[j] + 1 > dp[k]表示可以更新
                dp[k] = dp[j] + 1;
            }
        }
        ans = max(ans, dp[k]);
    }
    cout << ans << endl;
    return 0;
}