#pragma once

#include <ostream>
#include <type_traits>
#include <limits>

namespace core
{
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
    {
        os << "[ ";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            os << vec[i];
            if (i < vec.size()-1)
                os << ", ";
        }
        os << " ]";
        return os;
    }

    template<typename T>
    inline
    typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
    is_equal(const T& a, const T& b) noexcept
    {
        return a == b;
    }

    template<typename T>
    inline
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    is_equal(T a, T b) noexcept
    {
        T factor = max( static_cast<T>(1), max(std::fabs(a), std::fabs(b)) );
        return std::fabs(a - b) <= std::numeric_limits<T>::epsilon() * factor;
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
