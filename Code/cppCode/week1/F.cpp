#include <iostream>
using namespace std;


long long Solve(int x, int n) {
    long long ans = 1;// 初始只有一个动物
    for (int i = 0; i < n; i++){
        ans += ans * x;
    }
    return ans;
}
int main() {
    int x;// 每轮传染x个动物
    int n;// 轮数
    cin >> x >> n;
    long long ans = Solve(x, n);
    cout << ans;
    return 0;
}