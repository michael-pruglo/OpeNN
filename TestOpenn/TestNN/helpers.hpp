#pragma once

#include "../helpers.hpp"
#include "../../OpeNN/opeNN.hpp"

namespace openn
{
	void testNode(const Node& n, size_t inputs_count);
	void testLayer(const Layer& l, size_t prev_layer_size);
	void testNet(const NeuralNetwork& nn, const std::vector<size_t>& layer_sizes);
	

	class TestWithTestcases : public ::testing::Test
	{
	protected:
		virtual ~TestWithTestcases() = default;

		void SetUp() override;
		virtual void addRandCase() {}

		void runCases(size_t start, size_t finish);
		virtual void runCase(size_t i) {}
	};

}