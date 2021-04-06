#pragma once

#include <type_traits>
#include <limits>
#include <cmath>

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
}
