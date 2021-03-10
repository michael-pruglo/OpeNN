#pragma once

#include <TestOpenn/helpers.hpp>
#include <OpeNN/package/types.hpp>

namespace openn
{
	void test_nn_structure(const INeuralNetwork& nn, const std::vector<LayerMetadata>& nn_structure);

	template<typename T>
	std::vector<T> rand_param_vec(size_t n)
	{
		return core::generate(n, []{ return T::generateRand(); });
	}

	ActivationFType rand_activation();
}