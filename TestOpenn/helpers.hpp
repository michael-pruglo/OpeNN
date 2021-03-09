#pragma once

#include "gtest/gtest.h"
#include <Core/random.hpp>

inline void AssertInRange(double val, double min = 0.0, double max = 1.0)
{
	constexpr double EPS = 1e-9;
	ASSERT_GE(val, min-EPS);
	ASSERT_LE(val, max+EPS);
}

template<typename FloatT>
void AssertNear(const std::vector<FloatT>& v1, const std::vector<FloatT>& v2, FloatT abs_error)
{
	const size_t N = v1.size();
	ASSERT_EQ(v2.size(), N);

	for (size_t i = 0; i < N; ++i)
		ASSERT_NEAR(v1[i], v2[i], abs_error);
}

inline size_t rand_size(size_t max = 100)
{
	return static_cast<size_t>(core::randi(0, static_cast<int>(max)));
}

