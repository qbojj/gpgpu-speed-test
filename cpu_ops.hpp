#pragma once

#include <vector>

#include <cstdlib>

void cpu_add(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c);
void cpu_sub(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c);

void cpu_div_const(const std::vector<float> &a, float div, std::vector<float> &c);
void cpu_MA_const(const std::vector<float> &in, size_t MA, std::vector<float> &out);
void cpu_cpy(const std::vector<float> &a, std::vector<float> &b);