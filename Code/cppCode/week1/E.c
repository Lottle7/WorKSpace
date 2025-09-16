#include<stdio.h>

double values[100];
double p[5] = {0,0,0,0,0};
int ans[5] = {0,0,0,0,0};
int main()
{
    int n;
    scanf("%d", &n);
    for(int i = 0; i < n; i++){
        scanf("%lf", &values[i]);
    }
    int now = 0;
    for(int i = 0; i < n; i++){
        // 第一个人直接进队
        if (now == 0){
            p[0] = values[i];
            ans[0] = i+1;
            now++;
        }else if (values[i] > p[now-1] && now < 5){
            // 比队尾大且队伍未满 直接进队
            p[now] = values[i];
            ans[now] = i+1;
            now++;
        }else if (now == 5 && values[i] > p[4]){
            // 队列前移 排在队尾
            for (int j = 0; j < 4; j++)
            {
                ans[j] = ans[j+1];
                p[j] = p[j+1];
            }
            ans[4] = i+1;
            p[4] = values[i];
        }else{
            continue;
        }
    }    
    // 输出结果
    for (int i = 0; i < now; i++)
    {
        printf("%d ",ans[i]);
    }
    return 0;
}