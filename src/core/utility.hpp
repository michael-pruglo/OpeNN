#pragma once

#include <ostream>
#include <type_traits>
#include <limits>
#include <cmath>

namespace core
{
    namespace detail
    {
        template<typename, typename = void>
        struct is_iterable : std::false_type {};
        template<typename T>
        struct is_iterable<T, std::void_t<
            decltype(std::declval<T>().begin()),
            decltype(std::declval<T>().end()),
            decltype(++std::declval<decltype(std::declval<T&>().begin())&>()),
            decltype(std::declval<T>().begin() != std::declval<T>().end())
        >> : std::true_type {};
    }

    template<typename LinearContainer>
    constexpr
    typename std::enable_if_t<
        detail::is_iterable<LinearContainer>::value
        && !std::is_same_v<typename LinearContainer::value_type, char>,
    std::ostream&>
    operator<<(std::ostream& os, const LinearContainer& c)
    {
        os << "[ ";
        for (auto it = c.begin(); it != c.end(); )
        {
            os << *it;
            if (++it != c.end())
                os << ", ";
        }
        os << " ]";
        return os;
    }


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


    // generates an `std::vector<T>` of length `n`
    // generator is independent from the index: `v[i] = g()`
    template<typename T, typename Generator>
    std::vector<T> generate(size_t n, const Generator& g)
    {
        std::vector<T> res;
        res.reserve(n);
        for (size_t i = 0; i < n; ++i)
            res.emplace_back(g());
        return res;
    }

    // generates an `std::vector<T>` of length `n`
    // generator is dependent on the index: `v[i] = g(i)`
    template<typename T, typename Generator>
    std::vector<T> generate_i(size_t n, const Generator& g)
    {
        std::vector<T> res;
        res.reserve(n);
        for (size_t i = 0; i < n; ++i)
            res.emplace_back(g(i));
        return res;
    }

    template<typename MapFunc, typename T>
    std::vector<T> map(const MapFunc& map_func, const std::vector<T>& v)
    {
        return generate_i<T>(v.size(), [&](size_t i){ return map_func(v[i]); });
    }
}
