#include <stdio.h>

int values[200000];
// 用于存储每个200的余数出现的次数
int num[200] = {0};

int main() {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &values[i]);
        // 记录每个200的余数出现的次数
        num[values[i] % 200]++;
    }
    long long ans = 0;
    for (int i = 0; i < 200; i++) {
        // 计算组合数C(num[i], 2)
        if (num[i] > 1) {
            // 1ll 确保数字1被视为longlong类型
            // 组合数计算公式 C(n, 2) = n * (n - 1) / 2
            ans += 1LL * num[i] * (num[i] - 1) / 2;
        }
    }
    printf("%lld\n", ans);
    return 0;
}

/*
int values[200000];
int judge(int k){
    return k % 200 == 0;
}
int main() {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &values[i]);
    }
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
        {
            if(judge(values[i] - values[j]) == 1 && i != j){
                ans++;
            }else{
                continue;
            }
        }
        
    }
    printf("%lld\n", ans/2);
    return 0;
}
*/


