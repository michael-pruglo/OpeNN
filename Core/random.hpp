#pragma once

#include "types.hpp"
#include "utility.hpp"
#include <random>
#include <ctime>

namespace core
{
	inline std::random_device dev;
	inline std::mt19937 rnd_engine(dev() ^ static_cast<unsigned int>(time(nullptr)));

	inline float_t randd(float_t min = 0.0, float_t max = 1.0)
	{
		const std::uniform_real_distribution<float_t> distibution(min, max);
  		return distibution(rnd_engine);
	}

	inline int randi(int min, int max)
	{
		const std::uniform_int_distribution<int> distribution(min, max);
		return distribution(rnd_engine);
	}
	
	constexpr float_t W_MIN = -10.0, W_MAX = 10.0;

	inline Vec randVec(size_t n)
	{
		return core::generate<Vec>(n, []{ return randd(W_MIN, W_MAX); }); 
	}

	inline Matrix randMatrix(size_t n, size_t m)
	{
		return core::generate<Matrix>(n, [m]{ return randVec(m); }); 
	}
}
