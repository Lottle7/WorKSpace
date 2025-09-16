#include <stdio.h>

int a[100];
int b[100];
int main(){
    int n;
    scanf("%d", &n);
    for(int i = 0; i < n; i++){
        scanf("%d", &a[i]);
    }
    for (int i = 0;i < n; i++)
    {
        scanf("%d", &b[i]);
    }
    long long ans = 0;
    for(int x = 1; x <= 1000; x++){
        int flag = 1;
        for(int i = 0; i < n; i++){
            if(x < a[i] || x > b[i]){
                flag = 0;
                break;
            }
        }
        if(flag){
            ans++;
        }
    }


    printf("%lld\n", ans);
    return 0;
}