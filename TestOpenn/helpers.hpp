#pragma once
#include "pch.h"

inline void AssertInRange(double val, double min = 0.0, double max = 1.0)
{
	constexpr double EPS = 1e-9;
	ASSERT_GE(val, min-EPS);
	ASSERT_LE(val, max+EPS);
}