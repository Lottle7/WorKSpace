#include <bits/stdc++.h>
using namespace std;

// 整数位乘
struct BigBinary
{
    std::vector<int> x;     // 由低位到高位保存二进制位
    bool neg;               // 标记数的正负
    void Init(){x.clear(); neg = false;}
    BigBinary(){Init();}
    void Print()
    {
        if(neg && !x.empty()) printf("-");
        for(int i = x.size() - 1; i >= 0; i --)
            printf("%d", x[i]);
        if(x.empty()) printf("0");
        printf("\n");
    }
};
BigBinary Add(BigBinary &a, BigBinary &b)
{
    // TODO: 返回 a + b 的结果，小心两数异号情况
    BigBinary res(a);
    for(int i = 0; i < b.x.size(); i ++) res.x[i] += b.x[i] * b.neg ? -1 : 1;
    return res;

}
BigBinary Minus(BigBinary &a, BigBinary &b)
{
    // TODO: 返回 a - b 的结果，注意正负号
    
}
BigBinary Mul(BigBinary &a, BigBinary &b)
{
    // TODO: 模拟竖式计算两数相乘
}
void MulN2(BigBinary &a, int n_2)
{
    // TODO: 为 a 添加 n_2 个二进制0，即乘以 2^(n_2)
}
BigBinary FasterMul(BigBinary &a, BigBinary &b)
{
    // TODO: 分治优化位乘
    // 设 a = A * 2^(n/2) + B， b = C * 2^(n/2) + D
    // a*b = AC*2^n + [(A - B)(D - C) + AC + BD] * 2^(n/2) + BD
    // 注意：a 和 b 位数不一定相同，需要额外处理边界。
}

int main(){
    BigBinary a, b,ans;


    return 0;
}