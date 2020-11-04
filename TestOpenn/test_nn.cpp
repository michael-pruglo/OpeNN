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
			ASSERT_EQ(nn.layers[i].size(), layer_sizes[i]) << nn << layer_sizes;
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

	class NNAddLayerFixture : public ::testing::Test
	{
	protected:
		struct Testcase
		{
			size_t in;
			std::vector<size_t> insertions;
			size_t out;
		};

		std::vector<Testcase> testcases = {
			{ 0, {7}, 9 },
			{ 1, {0}, 9 },
			{ 3, {7}, 0 },
			{ 0, {0}, 3 },
			{ 7, {0}, 0 },
			{ 0, {7}, 0 },
			{ 0, {0}, 0 },
			{ 1, {1}, 1 },
			{ 2, {3}, 4 },
			{ 2, {3, 4}, 4 },
			{ 2, {9, 14}, 4 },
			{ 2, {9, 14}, 0 },
			{ 0, {9, 14}, 0 },
			{ 2, {0, 0}, 9 },
			{ 0, {0, 0}, 9 },
			{ 2, {0, 0}, 0 },
			{ 0, {0, 0}, 0 },
			{ 2, {7, 19, 20, 0, 1}, 4 },
			{ 2, {0, 0, 0, 0, 0, 0}, 4 },
		};

		const int RAND_CNT = 50;

		void SetUp() override
		{
			for (int i = 0; i < RAND_CNT; ++i)
			{
				std::vector<size_t> gen_ins(rand_int(1, 30));
				for (auto& ins: gen_ins) ins = rand_int(0, 20);
				testcases.push_back({
					static_cast<size_t>(rand_int(0, 20)),
					gen_ins, 
					static_cast<size_t>(rand_int(0, 20))
				});
			}
		}

		void runCases(size_t start, size_t finish)
		{
			for (size_t i = start; i < finish; ++i)
				runCase(i);
		}

		void runCase(size_t i)
		{
			const auto& tcas = testcases[i];
			NeuralNetwork nn(tcas.in, tcas.out);
			std::vector<size_t> curr_config = { tcas.in };
			for (const auto& ins: tcas.insertions)
			{
				nn.addLayer(ins);
				curr_config.push_back(ins);
			}
			curr_config.push_back(tcas.out);

			testNet(nn, curr_config);
		}
	};

	TEST_F(NNAddLayerFixture, AddLayerIns1Zeros1) { runCases(0, 3); }
	TEST_F(NNAddLayerFixture, AddLayerIns1Zeros2) { runCases(3, 6); }
	TEST_F(NNAddLayerFixture, AddLayerIns1Zeros3) { runCase(6); }
	TEST_F(NNAddLayerFixture, AddLayerIns1Regular) { runCases(7, 9); }
	TEST_F(NNAddLayerFixture, AddLayerIns2Regular) { runCases(9, 11); }
	TEST_F(NNAddLayerFixture, AddLayerIns2Zeros) { runCases(12, 17); }
	TEST_F(NNAddLayerFixture, AddLayerIns2ZerosAll) { runCase(17); }
	TEST_F(NNAddLayerFixture, AddLayerInsMultiple) { runCases(18, 20); }
	TEST_F(NNAddLayerFixture, AddLayerRnd) { runCases(20, testcases.size()); }
}