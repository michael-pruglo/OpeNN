#pragma once

#include <random>
#include <ctime>
#include "opeNN.hpp"

namespace openn
{
	inline std::random_device dev;
	inline std::mt19937 rnd_engine(dev() ^ static_cast<unsigned int>(time(nullptr)));

	inline float_t randd(float_t min = 0.0, float_t max = 1.0)
	{
		const std::uniform_real_distribution<float_t> distibution(min, max);
  		return distibution(rnd_engine);
	}


	struct ActivationF
	{
		static float_t ReLU		(float_t x) { return std::max(0., x); } 
		static float_t sigmoid	(float_t x) { return 1. / (1. + std::exp(-x)); }
		static float_t softplus	(float_t x) { return std::log(1. + std::exp(x)); }
		static float_t tanh		(float_t x) { return std::tanh(x); }
	};
}
