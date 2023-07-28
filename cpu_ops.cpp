#include "cpu_ops.hpp"


void cpu_add(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c)
{
    for( size_t i = 0; i < a.size(); i++ )
        c[i] = a[i] + b[i];
}

void cpu_sub(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c)
{
    for( size_t i = 0; i < a.size(); i++ )
        c[i] = a[i] - b[i];
}

void cpu_div_const(const std::vector<float> &a, float div, std::vector<float> &c)
{
    for( size_t i = 0; i < a.size(); i++ )
        c[i] = a[i] / div;
}

void cpu_MA_const(const std::vector<float> &in, size_t MA, std::vector<float> &out)
{
    float acc = 0.f;
    size_t len = in.size();

    for( size_t i = 0; i < std::min(MA, len); ++i )
    {
        acc += in[i];
        out[i] = acc / MA;
    }

    for( size_t i = MA; i < len; ++i )
    {
        acc += in[i];
        acc -= in[i-MA];
        out[i] = acc / MA;
    }
}

void cpu_cpy(const std::vector<float> &a, std::vector<float> &b)
{
    b = a;
}
