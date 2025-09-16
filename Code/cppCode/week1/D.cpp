#include <iostream>
using namespace std;


int num[100];
char flag[100];
int main(){
    int n;
    cin >> n;
    for(int i = 0; i < n; i++){
        cin >> num[i];
    }
    for(int i = 0; i < n -1; i++){
        cin >> flag[i];
    }
    long long ans = num[0];
    for(int i = 0; i < n - 1; i++){
        if(flag[i] == '+') ans += num[i + 1];
        else if(flag[i] == '-') ans -= num[i + 1];
        else if(flag[i] == '*') ans *= num[i + 1];
    }
    cout << ans;
    return 0;
}