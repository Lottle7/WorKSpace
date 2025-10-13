#include <iostream>
#include <vector>
using namespace std;


// 用栈的思想解决运算符的顺序问题
int num[100];
char flag[100];
int main(){
    int n;
    cin >> n;
    for(int i = 0; i < n; i++){
        cin >> num[i];
    }
    for(int i = 0; i < n - 1; i++){
        cin >> flag[i];
    }
    // vector容器 push_back 添加到vector的末尾
    vector<long long> stack;
    stack.push_back(num[0]);
    for(int i = 0; i < n - 1; i++){
        // 遇到*就出栈直接进行计算
        if(flag[i] == '*'){
            // 取栈顶元素
            long long end = stack.back();
            // 弹栈
            stack.pop_back();
            // 刚取出的元素与下一个元素相乘并入栈
            stack.push_back( end * num[i+1]);
        } else if(flag[i] == '+') {
            // 直接入栈
            stack.push_back(num[i + 1]);
        } else if(flag[i] == '-') {
            // 如果是-号 取负再入栈
            stack.push_back(-num[i + 1]);
        }
    }
    long long ans = 0;
    // 所有栈内元素都出闸并相加
    for(auto v : stack) ans += v;
    cout << ans;
    return 0;
}