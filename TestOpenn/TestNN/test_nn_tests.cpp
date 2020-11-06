#include "test_nn_instantiations.hpp"
#include "../../packages/nlohmann/json.hpp"
#include "../../OpeNN/package/io/nn_serializer.cpp"

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
			NeuralNetwork(param.init_in, param.init_out), 
			{ param.init_in, param.init_out }
		);
	}



	TEST_P(NNAddLayerFixture, AddsLayer)
	{
		const auto& param = GetParam();
		testNet(param.createNN(), param.expectedResultSizes());
	}



	TEST_P(NNJsonSerializeFixture, SerializesDeserializes)
	{
		const auto& nn1 = GetParam().createNN();
		const nlohmann::json nn1_json = nn1;
		const NeuralNetwork nn2 = nn1_json;
		ASSERT_EQ(nn1, nn2);
	}
}
