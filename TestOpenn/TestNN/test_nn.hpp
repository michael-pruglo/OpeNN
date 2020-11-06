#pragma once

#include "helpers.hpp"

namespace openn
{
	struct ConstructNNParam
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

		static ConstructNNParam generateRand();
	};

	class NNConstructFixture		: public testing::TestWithParam<ConstructNNParam> {};
	class NNAddLayerFixture			: public testing::TestWithParam<ConstructNNParam> {};
	class NNJsonSerializeFixture	: public testing::TestWithParam<ConstructNNParam> {};
}