#include <iostream>
#include <cstring>
using namespace std;


int n,b;
int w[1005];
int v[10005];
int dp[1005][10005]; //dp[i][j]表示前i个物品在容量j下的最大价值
int main(){
    cin >> n >> b;
    for(int i = 1;i <= n;i++){
        cin >> w[i] >> v[i];
    }
    memset(dp,0,sizeof(dp));
    for(int i = 1; i <=n; i++){
        for(int j = 0; j <=b; j++){
            dp[i][j] = dp[i-1][j];// 用前i-1个物品的最大价值初始化
            if(w[i] <= j){
                dp[i][j] = max(dp[i-1][j],dp[i][j-w[i]] + v[i]);
            }
        }
    }
    cout << dp[n][b] << endl;
    return 0;
}