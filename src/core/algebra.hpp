#pragma once

#include <core/types.hpp>

namespace core
{
    Vec sigmoid (const Vec& x);
    Vec relu    (const Vec& x);
    Vec softplus(const Vec& x);
    Vec tanh    (const Vec& x);

    Vec der_sigmoid (const Vec& x);
    Vec der_relu    (const Vec& x);
    Vec der_softplus(const Vec& x);
    Vec der_tanh    (const Vec& x);

    //`a` = given, `y` = expected
    float_t mean_squared_eror(const Vec& a, const Vec& y);
    float_t cross_entropy    (const Vec& a, const Vec& y);

    float_t der_mean_squared_eror(const Vec& a, const Vec& y);
    float_t der_cross_entropy    (const Vec& a, const Vec& y);

    //hadamard product is element-wise prod of two vectors
}