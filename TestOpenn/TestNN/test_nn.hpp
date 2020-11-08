#pragma once

#include <TestOpenn/TestNN/helpers.hpp>

namespace openn
{
	struct ConstructNNParam
	{
		struct Insertion {
			explicit Insertion(size_t layer_size, ActivationFType activation = ActivationFType::sigmoid);
			Insertion(size_t layer_size, size_t pos, ActivationFType activation = ActivationFType::sigmoid);

			size_t layer_size, pos; 
			bool use_pos;
			ActivationFType activation;
		
			static Insertion generateRand(size_t max_allowed_pos);
		};

		[[nodiscard]] std::vector<size_t> expectedResultSizes() const;
		[[nodiscard]] NeuralNetwork createNN() const;

		size_t init_in, init_out;
		std::vector<Insertion> insertions;

		static ConstructNNParam generateRand();
	};

	class NNConstructFixture	: public testing::TestWithParam<ConstructNNParam> {};
	class NNAddLayerFixture		: public testing::TestWithParam<ConstructNNParam> {};
	class NNjsonFixture			: public testing::TestWithParam<ConstructNNParam> {};


	struct InpOutpNNParam
	{
		std::vector<float_t> in, out;
		float_t abs_error;
		std::string filename;
	};

	class NNForwardFixture		: public testing::TestWithParam<InpOutpNNParam> {};
}