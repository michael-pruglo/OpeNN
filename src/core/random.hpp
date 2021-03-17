#pragma once

#include <core/types.hpp>
#include <core/utility.hpp>
#include <random>
#include <ctime>
#include <type_traits>

namespace core
{
    inline std::random_device dev;
    inline std::mt19937 rnd_engine(dev() ^ static_cast<unsigned int>(time(nullptr)));


    template<typename Floating = float_t>
    inline
    typename std::enable_if_t<std::is_floating_point_v<Floating>, Floating>
    rand_d(Floating min = 0.0, Floating max = 1.0)
    {
        std::uniform_real_distribution<Floating> distibution(min, max);
        return distibution(rnd_engine);
    }

    template<typename Integral = int>
    inline
    typename std::enable_if_t<!std::is_floating_point_v<Integral>, Integral>
    rand_i(Integral min, Integral max)
    {
        std::uniform_int_distribution<Integral> distribution(min, max);
        return distribution(rnd_engine);
    }


    inline Vec rand_vec(size_t n, float_t min = -10.0, float_t max = 10.0)
    {
        return core::generate(n, [min, max]{ return rand_d(min, max); });
    }

    inline Matrix rand_matrix(size_t n, size_t m)
    {
        return core::generate(n, [m]{ return rand_vec(m); });
    }
}
