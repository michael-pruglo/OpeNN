#pragma once
#include "gtest/gtest.h"

inline void AssertInRange(double val, double min = 0.0, double max = 1.0)
{
	constexpr double EPS = 1e-9;
	ASSERT_GE(val, min-EPS);
	ASSERT_LE(val, max+EPS);
}

inline int rand_int(int min, int max)
{
	return rand()%(max-min+1)+min;
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