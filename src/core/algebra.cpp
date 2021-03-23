#include <core/algebra.hpp>
#include <core/utility.hpp>
#include <cassert>
#include <cmath>
#include <numeric>

using namespace core;

core::float_t core::sigmoid(float_t x)
{
    return 1. / (1. + std::exp(-x));
}
core::float_t core::relu(float_t x)
{
    return std::max(0., x);
}
core::float_t core::softplus(float_t x)
{
    return std::log(1. + std::exp(x));
}
core::float_t core::tanh(float_t x)
{
    return std::tanh(x);
}

core::float_t core::der_sigmoid(float_t x)
{
    return sigmoid(x)*(1. - sigmoid(x));
}
core::float_t core::der_relu(float_t x)
{
    return x >= 0.;
}
core::float_t core::der_softplus(float_t x)
{
    return 1. / (1. + std::exp(-x));
}
core::float_t core::der_tanh(float_t x)
{
    return 1. - std::pow(std::tanh(x), 2);
}

core::float_t core::mean_squared_eror(const Vec& v, const Vec& exp)
{
    assert(v.size() == exp.size());

    float_t c = 0.;
    for (size_t i = 0; i < exp.size(); ++i)
        c += std::pow(v[i]-exp[i], 2);
    return c;
}

core::float_t core::cross_entropy(const Vec &v, const Vec &exp)
{
    assert(v.size() == exp.size());
    return 0.;
}

Vec core::operator*(const Matrix& m, const Vec& v)
{
    assert(m.cols() == v.size());

    return generate_i(m.rows(), [&](size_t i) {
        return std::inner_product(m[i].begin(), m[i].end(), v.begin(), 0.);
    });
}

Vec core::operator+(const Vec& v1, const Vec& v2)
{
    assert(v1.size() == v2.size());
    
    return generate_i(v1.size(), [&](size_t i) {
        return v1[i] + v2[i];
    });
}
