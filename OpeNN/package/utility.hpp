#pragma once

#include <random>
#include <ctime>
#include <iostream>

namespace openn
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

}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
	os << "[ ";
	for (size_t i = 0; i < vec.size()-1; ++i)
		os << vec[i] << ", ";
	os << vec.back() << " ]";
	return os;
}