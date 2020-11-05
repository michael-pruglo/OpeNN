#pragma once

#include "../helpers.hpp"
#include "../../OpeNN/opeNN.hpp"

namespace openn
{
	void testNode(const Node& n, size_t inputs_count);
	void testLayer(const Layer& l, size_t prev_layer_size);
	void testNet(const NeuralNetwork& nn, const std::vector<size_t>& layer_sizes);

	template<typename Param>
	std::vector<Param> generateRandCtorParam(size_t n)
	{
		std::vector<Param> res;
		res.reserve(n);
		for (size_t i = 0; i < n; ++i)
			res.push_back(Param::generateRand());
		return res;
	}
}