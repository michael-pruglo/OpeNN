#include "test_nn_instantiations.hpp"

namespace openn
{
	TEST(NodeTest, Constructors)
	{
		for (size_t i = 0; i < 100; ++i)
			testNode( Node(i), i );
	}
	


	TEST_P(NNConstructFixture, Constructs)
	{
		const auto& param = GetParam();
		testNet(
			NeuralNetwork(param.in_size, param.out_size), 
			{ param.in_size, param.out_size }
		);
	}



	TEST_P(NNAddLayerFixture, AddsLayer)
	{
		const auto& param = GetParam();
		testNet(param.createNN(), param.expectedResultSizes());
	}
}