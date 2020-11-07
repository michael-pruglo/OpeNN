#pragma once

#include <OpeNN/package/opeNN.hpp>
#include <random>
#include <ctime>

namespace openn
{
	inline std::random_device dev;
	inline std::mt19937 rnd_engine(dev() ^ static_cast<unsigned int>(time(nullptr)));

	inline float_t randd(float_t min = 0.0, float_t max = 1.0)
	{
		const std::uniform_real_distribution<float_t> distibution(min, max);
  		return distibution(rnd_engine);
	}
}
