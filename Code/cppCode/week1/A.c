#include<stdio.h>

int main() {
    // 开longlong保证精度
    long long N;
    scanf("%lld", &N);
    int k = 0;
    // 位运算求解
    while ((1LL << (k + 1)) <= N) {
        k++;
    }
    printf("%d\n", k);
    return 0;
}