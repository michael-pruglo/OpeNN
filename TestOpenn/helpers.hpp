#pragma once

#include "gtest/gtest.h"
#include <random>

inline void AssertInRange(double val, double min = 0.0, double max = 1.0)
{
	constexpr double EPS = 1e-9;
	ASSERT_GE(val, min-EPS);
	ASSERT_LE(val, max+EPS);
}

inline std::random_device dev;
inline std::mt19937 rnd_engine(dev() ^ static_cast<unsigned int>(time(nullptr)));

inline int rand_int(int min, int max)
{
	const std::uniform_int_distribution<int> distibution(min, max);
  	return distibution(rnd_engine);
}

inline size_t rand_size(size_t max = 100)
{
	return static_cast<size_t>(rand_int(0, max));
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
