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
    Vec mean_squared_eror(const Vec& a, const Vec& y);
    Vec cross_entropy    (const Vec& a, const Vec& y);

    Vec der_mean_squared_eror(const Vec& a, const Vec& y);
    Vec der_cross_entropy    (const Vec& a, const Vec& y);

    //hadamard product is element-wise prod of two vectors
}