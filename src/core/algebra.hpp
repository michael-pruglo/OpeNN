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

    //hadamard product is element-wise prod of two vectors
}