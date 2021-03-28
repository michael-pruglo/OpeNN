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

    //`a` = given, `y` = expected
    float_t mean_squared_eror(const Vec& a, const Vec& y);
    float_t cross_entropy    (const Vec& a, const Vec& y);

    //Matrix algebra
    Vec     operator*(const Matrix& m,  const Vec& v);
    Matrix  operator+(const Matrix& m1, const Matrix& m2);
    Matrix  operator/(const Matrix& m,  float_t divisor);

    //Vec algebra
    Vec     operator+(const Vec& v1, const Vec& v2);
    Vec     operator/(const Vec& v1, float_t divisor);
    Vec     hadamard (const Vec& v1, const Vec& v2);
}