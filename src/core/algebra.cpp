#include <core/algebra.hpp>
#include <cassert>
#include <cmath>

using namespace core;

core::Vec core::sigmoid(const Vec& x)
{
    return 1. / (1. + xt::exp(-x));
}
core::Vec core::relu(const Vec& x)
{
    Vec zeros = xt::zeros<float_t>({x.size()});
    return xt::maximum(zeros, x);
}
core::Vec core::softplus(const Vec& x)
{
    return xt::log(1. + xt::exp(x));
}
core::Vec core::tanh(const Vec& x)
{
    return xt::tanh(x);
}

core::Vec core::der_sigmoid(const Vec& x)
{
    return sigmoid(x)*(1. - sigmoid(x));
}
core::Vec core::der_relu(const Vec& x)
{
    return x >= 0.;
}
core::Vec core::der_softplus(const Vec& x)
{
    return 1. / (1. + xt::exp(-x));
}
core::Vec core::der_tanh(const Vec& x)
{
    return 1. - xt::pow(xt::tanh(x), 2);
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
