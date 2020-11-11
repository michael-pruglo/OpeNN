#include "algebra.hpp"
#include "utility.hpp"
#include <cassert>
#include <cmath>
#include <numeric>

using namespace core;

core::float_t core::norm_diff(const Vec& v1, const Vec& v2)
{
	assert(v1.size() == v2.size());

	float_t c = 0;
	for (size_t i = 0; i < v2.size(); ++i)
		c += std::pow(v1[i] - v2[i], 2);
	return c;
}

Vec core::operator*(const Matrix& m, const Vec& v)
{
	return generate_i<Vec>(v.size(), [&](size_t i) { 
		return std::inner_product(m[i].begin(), m[i].end(), v.begin(), 0.); 
	});
}

Vec core::operator+(const Vec& v1, const Vec& v2)
{
	return generate_i<Vec>(v1.size(), [&](size_t i) {
		return v1[i] + v2[i];
	});
}