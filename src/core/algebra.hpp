#pragma once

#include <core/types.hpp>

namespace core
{
    float_t norm_diff(const Vec& v1, const Vec& v2);

    Vec operator*(const Matrix& m, const Vec& v);
    Vec operator+(const Vec& v1, const Vec& v2);
}