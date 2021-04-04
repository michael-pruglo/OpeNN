#pragma once

#include <ostream>
#include <type_traits>
#include <limits>
#include <cmath>
#include <xtensor/xtensor.hpp>

namespace core
{
    template<typename T>
    constexpr
    typename std::enable_if_t<!std::is_floating_point_v<T>, bool>
    is_equal(const T& a, const T& b) noexcept
    {
        return a == b;
    }

    template<typename T>
    constexpr
    typename std::enable_if_t<std::is_floating_point_v<T>, bool>
    is_equal(T a, T b, T tolerance = static_cast<T>(0)) noexcept
    {
        T factor = std::max( static_cast<T>(1), std::max(std::fabs(a), std::fabs(b)) );
        return std::fabs(a - b) <= tolerance + std::numeric_limits<T>::epsilon() * factor;
    }


    // generates a vector of length `n`
    // generator is independent from the index: `v[i] = g()`
    template<typename Generator>
    auto generate(size_t n, const Generator& g)
    {
        using Tensor = xt::xtensor<decltype(g()), 1>;
        typename Tensor::shape_type shape = {n};
        Tensor res(shape);
        for (size_t i = 0; i < n; ++i)
            res[i] = g();
        return res;
    }

    // generates a vector of length `n`
    // generator is dependent on the index: `v[i] = g(i)`
    template<typename Generator>
    auto generate_i(size_t n, const Generator& g)
    {
        using Tensor = xt::xtensor<decltype(g(0)), 1>;
        typename Tensor::shape_type shape = {n};
        Tensor res(shape);
        for (size_t i = 0; i < n; ++i)
            res[i] = g(i);
        return res;
    }

    template<typename MapFunc, typename Container>
    auto map(const MapFunc& map_func, const Container& v)
    {
        return generate_i(v.size(), [&](size_t i){ return map_func(v[i]); });
    }
}
