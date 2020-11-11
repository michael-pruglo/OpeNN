#pragma once

#include "types.hpp"
#include <iostream>

namespace core
{
	template<typename T>
	std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
	{
		os << "[ ";
		for (size_t i = 0; i < vec.size()-1; ++i)
			os << vec[i] << ", ";
		os << vec.back() << " ]";
		return os;
	}

	inline bool float_eq(float_t f1, float_t f2, float_t EPS = 1e-9)
	{
		return std::abs(f1-f2) < EPS;
	}
	
	template<typename Vector, typename Generator>
	Vector generate(size_t n, const Generator& g)
	{
		Vector res;
		res.reserve(n);
		for (size_t i = 0; i < n; ++i)
			res.emplace_back(g());
		return res;
	}
	
	template<typename Vector, typename Generator>
	Vector generate_i(size_t n, const Generator& g)
	{
		Vector res;
		res.reserve(n);
		for (size_t i = 0; i < n; ++i)
			res.emplace_back(g(i));
		return res;
	}

	template<typename Vector, typename MapFunc>
	Vector map(const Vector& v, const MapFunc& map_func)
	{
		return generate_i<Vector>(v.size(), [&](size_t i){ return map_func(v[i]); } );
	}


}
