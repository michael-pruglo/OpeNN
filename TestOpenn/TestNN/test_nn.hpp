#pragma once

#include "helpers.hpp"

namespace openn
{
	struct ConstructNNParam
	{
		size_t in_size, out_size;

		static ConstructNNParam generateRand();
	};

	class NNConstructFixture :
		public testing::TestWithParam<ConstructNNParam>
	{};




	struct AddLayerTestParam
	{
		struct Insertion {
			explicit Insertion(size_t layer_size);
			Insertion(size_t layer_size, size_t pos);

			size_t layer_size, pos; 
			bool use_pos; 
		
			static Insertion generateRand(size_t max_allowed_pos);
		};

		[[nodiscard]] std::vector<size_t> expectedResultSizes() const;
		[[nodiscard]] NeuralNetwork createNN() const;

		size_t init_in, init_out;
		std::vector<Insertion> insertions;

		static AddLayerTestParam generateRand();
	};

	class NNAddLayerFixture :
		public testing::TestWithParam<AddLayerTestParam>
	{};
}