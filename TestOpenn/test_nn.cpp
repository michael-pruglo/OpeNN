#include "pch.h"
#include "helpers.hpp"
#include "../OpeNN/opeNN.hpp"
#include "../OpeNN/opeNN.cpp"

namespace openn
{
	void testNode(const Node& n, size_t inputs_count)
	{
		AssertInRange(n.val);
		AssertInRange(n.b);
		ASSERT_EQ(n.w.size(), inputs_count);
		for (const auto& weight: n.w)
			AssertInRange(weight);
	}

	void testLayer(const Layer& l, size_t prev_layer_size)
	{
		for (const auto& n : l)
			testNode(n, prev_layer_size);
	}

	void testNet(const NeuralNetwork& nn, const std::vector<size_t>& layer_sizes)
	{
		const size_t N = layer_sizes.size();
		ASSERT_EQ(nn.layers.size(), N);

		for (size_t i = 0; i < N; ++i)
		{
			ASSERT_EQ(nn.layers[i].size(), layer_sizes[i]);
			testLayer(nn.layers[i], i ? layer_sizes[i-1] : 0);
		}
	}

	TEST(NodeTest, Constructors)
	{
		for (size_t i = 0; i < 100; ++i)
			testNode( Node(i), i );
	}

	TEST(NeuralNetworkTest, Constructor)
	{
		const size_t N = 10;
		for (size_t i = 0; i < N; ++i)
			for (size_t j = 0; j < N; ++j)
			{
				NeuralNetwork nn(i, j);
				testNet(nn, {i, j});
			}
	}
}