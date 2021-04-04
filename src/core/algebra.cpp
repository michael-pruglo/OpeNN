#include <core/algebra.hpp>
#include <cassert>
#include <cmath>

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

core::float_t core::mean_squared_eror(const Vec& a, const Vec& y)
{
    assert(a.size() == y.size());

    float_t cost = 0.;
    for (size_t i = 0; i < y.size(); ++i)
        cost += std::pow(a[i]-y[i], 2);
    return cost;
}

core::float_t core::cross_entropy(const Vec& a, const Vec& y)
{
    assert(a.size() == y.size());

    float_t cost = 0.;
    for (size_t i = 0; i < y.size(); ++i)
        cost += -( y[i]*log(a[i]) + (1.-y[i])*log(1.-a[i]) );
    return cost;
}
