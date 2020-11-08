#pragma once

#include <OpeNN/package/types.hpp>
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

	inline bool float_eq(float_t f1, float_t f2, float_t EPS = 1e-9)
	{
		return std::abs(f1-f2) < EPS;
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

template<typename T, typename Generator>
void generative_append(std::vector<T>& v, size_t extra_amount, const Generator& gen)
{
	v.reserve(v.size() + extra_amount);
	for (size_t i = 0; i < extra_amount; ++i)
		v.emplace_back(gen());
}

template<typename T, typename Generator>
void generative_construct(std::vector<T>& v, size_t size, const Generator& gen)
{
	generative_append(v, size, gen);
}

template<typename T, typename Generator>
void generative_resize(std::vector<T>& v, size_t amount, const Generator& gen)
{
	if (v.size() > amount)
		v.resize(amount);
	else
		generative_append(v, amount - v.size(), gen);
}