#include <iostream>
#include <algorithm>
#include <cstring>
#include <math.h>
using namespace std;

char a[1005], b[1005];
int dp[1005][1005];//dp[i][j]表示a的前i个字符和b的前j个字符的最长公共子序列长度
int ans = 0;
int main(){
    while(scanf("%s%s", a + 1, b + 1) != EOF){
        memset(dp,0,sizeof(dp));
        ans = 0;
        for(int i = 1; a[i]; i++){
            for(int j = 1; b[j]; j++){
                if(a[i] == b[j]){
                    dp[i][j] = dp[i-1][j-1] + 1;
                }else{
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
                }
                ans = max(ans,dp[i][j]);
            }
        }
        cout << ans << endl;
    }
    return 0;
}