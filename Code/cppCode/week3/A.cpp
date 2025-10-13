#include <iostream>
using namespace std;

int n;
int table[32][32];
void dp(int n){
    if(n == 1){
        table[0][0] = 0;
         return;
    }
    // 先处理左上角
    dp(n/2);
   
    // 再处理右下角 直接把左上角copy到右下角
    for(int i =0;i< n/2;i++){
        for(int j =0;j< n/2;j++){
            // 右下角
            table[i+n/2][n/2+j] = table[i][j];
            // 右上角
            table[i][j+n/2] = table[i][j] + n/2;
            // 左下角
            table[n/2+i][j] = table[i][j] + n/2;
        }
    }
    
    return;

}

int main(){
    cin >> n;
    dp(n);
    for (int i = 0;i < n;i++){
        // 去掉第一列
        for(int j = 1; j<n;j++){
            // 输出要加1
            cout << table[i][j] + 1 << " ";
        }
        cout << endl;
    }    
    return 0;
}