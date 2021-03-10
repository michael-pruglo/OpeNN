#pragma once

#include <TestOpenn/helpers.hpp>
#include <OpeNN/package/types.hpp>

namespace openn
{
	void test_nn_structure(const INeuralNetwork& nn, const std::vector<LayerMetadata>& nn_structure);

	template<typename Param>
	std::vector<Param> rand_param_vec(size_t n)
	{
		return core::generate(n, []{ return Param::generateRand() });
	}

	ActivationFType rand_activation();
}