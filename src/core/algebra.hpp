#pragma once

#include <core/types.hpp>

namespace core
{
    float_t sigmoid (float_t x);
    float_t relu    (float_t x);
    float_t softplus(float_t x);
    float_t tanh    (float_t x);

    float_t der_sigmoid (float_t x);
    float_t der_relu    (float_t x);
    float_t der_softplus(float_t x);
    float_t der_tanh    (float_t x);

    float_t mean_squared_eror(const Vec& v, const Vec& exp);
    float_t cross_entropy    (const Vec& v, const Vec& exp);

    Vec operator*(const Matrix& m, const Vec& v);
    Vec operator+(const Vec& v1, const Vec& v2);
}