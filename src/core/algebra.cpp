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
    return 1. - xt::pow<2>(xt::tanh(x));
}

namespace
{
    inline core::float_t eval_sum(const Vec& v)
    {
        return xt::sum(v)[0];
    }
}

Vec core::mean_squared_eror(const Vec& a, const Vec& y)
{
    assert(a.size() == y.size());

    const auto& c = xt::pow<2>(a-y);

    return c;
}

Vec core::cross_entropy(const Vec& a, const Vec& y)
{
    assert(a.size() == y.size());

    const auto& ones = xt::ones<float_t>({a.size()});
    const auto& c = -( y*xt::log(a) + (ones-y)*xt::log(ones-a));

    return c;
}


Vec core::der_mean_squared_eror(const Vec& a, const Vec& y)
{
    assert(a.size() == y.size());

    const auto& c = 2 * (a-y);

    return c;
}

Vec core::der_cross_entropy(const Vec& a, const Vec& y)
{
    assert(a.size() == y.size());

    throw "not implemented";

    //const auto& c = a;

    //return eval_sum(c);
}