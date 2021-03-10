#pragma once

#include <Core/types.hpp>
#include <Core/utility.hpp>
#include <random>
#include <ctime>

namespace core
{
	inline std::random_device dev;
	inline std::mt19937 rnd_engine(dev() ^ static_cast<unsigned int>(time(nullptr)));
	
	
	
	template<typename Floating = float_t>
    inline
    typename std::enable_if_t<std::is_floating_point_v<Floating>, Floating>
	randd(Floating min = 0.0, Floating max = 1.0)
	{
		const std::uniform_real_distribution<Floating> distibution(min, max);
  		return distibution(rnd_engine);
	}

	template<typename Integral = int>
    inline
    typename std::enable_if_t<!std::is_floating_point_v<Integral>, Integral>
	randi(Integral min, Integral max)
	{
		const std::uniform_int_distribution<Integral> distribution(min, max);
		return distribution(rnd_engine);
	}
	


	constexpr float_t _W_MIN = -10.0, _W_MAX = 10.0;
	inline Vec randVec(size_t n)
	{
		return core::generate<float_t>(n, []{ return randd(_W_MIN, _W_MAX); }); 
	}

	inline Matrix randMatrix(size_t n, size_t m)
	{
		return core::generate<Vec>(n, [m]{ return randVec(m); }); 
	}
}
