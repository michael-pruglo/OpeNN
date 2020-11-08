#pragma once

#include <TestOpenn/TestNN/helpers.hpp>
#include <OpeNN/package/opeNN.hpp>

namespace openn
{
	struct ConstructNNParam
	{
		struct InsLayer 
		{
			explicit InsLayer(size_t layer_size, ActivationFType activation = ActivationFType::sigmoid);
			InsLayer(size_t layer_size, size_t pos, ActivationFType activation = ActivationFType::sigmoid);

			size_t layer_size, pos; 
			bool use_pos;
			ActivationFType activation;
		
			static InsLayer generateRand(size_t max_allowed_pos);
		};

		ConstructNNParam(std::vector<LayerStructure> nn_structure, std::vector<InsLayer> additional_insertions = {});

		[[nodiscard]] std::vector<LayerStructure> expectedResultStructure() const;
		[[nodiscard]] NeuralNetwork createNN() const;

		std::vector<LayerStructure> nn_structure;
		std::vector<InsLayer> additional_insertions;

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