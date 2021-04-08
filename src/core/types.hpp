#pragma once

#include <xtensor/xtensor.hpp>

namespace core
{
    using float_t   = double;
    using Vec       = xt::xtensor<float_t, 1>;
    using Vectors   = xt::xtensor<Vec, 1>;
    using Matrix    = xt::xtensor<float_t, 2>;
    using Matrixes  = xt::xtensor<Matrix, 1>;
}
