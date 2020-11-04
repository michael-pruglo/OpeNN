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
		std::vector<std::pair<size_t, size_t>> testcases = {
			{ 0, 0 },
			{ 0, 1 },
			{ 1, 0 },
			{ 1, 1 },
			{ 1, 2 },
			{ 1, 100 },
			{ 100, 1 },
			{ 100, 100 },
			{ 5, 1 },
			{ 1, 5 },
		};
		for (int i = 0; i < 50; ++i)
			testcases.emplace_back(rand_int(0,100), rand_int(0,100));


		for (const auto& [i, j] : testcases)
		{
			NeuralNetwork nn(i, j);
			testNet(nn, {i, j});
		}
	}

	TEST(NeuralNetworkTest, AddLayer)
	{
		struct Testcase
		{
			size_t in;
			std::vector<size_t> insertions;
			size_t out;
		};

		std::vector<Testcase> testcases = {
			{ 0, {7}, 9 },
		};

		for (const auto& tcas: testcases)
		{
			NeuralNetwork nn(tcas.in, tcas.out);
			std::vector<size_t> curr_config = { tcas.in, tcas.out };
			for (const auto& ins: tcas.insertions)
			{
				nn.addLayer(ins);
				curr_config.pop_back();
				curr_config.push_back(ins);
				curr_config.push_back(tcas.out);
				testNet(nn, curr_config);
			}
		}
	}
}