#include <iostream>
using namespace std;
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<algorithm>
const int maxn = 1e5 + 10;
int n, a[maxn], dp[maxn];
int Solve(int a[], int n, int dp[])
{
    // TODO: 计算最大子串和
}
int main()
{
    while(scanf("%d", &n) != EOF)
    {
        for(int i = 1; i <= n; i ++)
            scanf("%d", &a[i]);
        printf("%d\n", Solve(a, n, dp));
    }
    return 0;
}