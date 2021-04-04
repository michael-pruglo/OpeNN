#pragma once

#include <xtensor/xtensor.hpp>
#include <core/random.hpp>

namespace core
{
    using float_t = double;
    using Vec     = xt::xtensor<float_t, 1>;
    using Matrix  = xt::xtensor<float_t, 2>;

    template<typename Tensor>
    Tensor rand_tensor(typename Tensor::shape_type shape, float_t min = 0.0, float_t max = 1.0)
    {
        Tensor res(shape);
        for (auto& x: res)
            x = rand_d(min, max);
        return res;
    }
}
