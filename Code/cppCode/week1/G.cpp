#include <iostream>
using namespace std;

// n级台阶的情况可由 n-k ~ n-1级台阶的情况推导出来
// dp[n] = dp[n-1] + dp[n-2] + ... + dp[n-k]
int dp[1000007] = {0};
int main(){
    int n, k;
    cin >> n >> k;
    dp[0] = 1;
    for(int i = 1; i <= n; i++){
        if ( i - k < 0 ){
            for (int j = 0; j <= i-1; j++){
                dp[i] = (dp[i] + dp[j]) % 100003;
            }

        }else{
            for(int j = i -k; j <= i - 1; j++){
            dp[i] = (dp[i] + dp[j]) % 100003;
        }
        }
    }
    cout << dp[n];
    return 0;
}