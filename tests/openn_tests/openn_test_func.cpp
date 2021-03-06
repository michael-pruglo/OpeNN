//#include <OpeNN/package/io/nn_serializer.cpp>
//#include <packages/nlohmann/json.hpp>
//#include <TestOpenn/TestNN/test_nn_instantiations.hpp>
//
//namespace openn
//{
//	TEST(NodeTest, Constructors)
//	{
//		for (size_t i = 0; i < 100; ++i)
//			testNode( Node(i), i );
//	}
//
//	TEST_P(NNConstructFixture, Constructs)
//	{
//		const auto& param = GetParam();
//		testNet(
//			NeuralNetwork(param.nn_structure),
//			param.nn_structure
//		);
//	}
//
//	TEST_P(NNAddLayerFixture, AddsLayer)
//	{
//		const auto& param = GetParam();
//		testNet(param.createNN(), param.expectedResultStructure());
//	}
//
//	TEST_P(NNjsonFixture, SerializesDeserializes)
//	{
//		const auto& nn1 = GetParam().createNN();
//		const nlohmann::json nn1_json = nn1;
//		const NeuralNetwork nn2 = nn1_json;
//		ASSERT_EQ(nn1, nn2);
//	}
//
//	TEST_P(NNjsonFixture, ToFile)
//	{
//		const auto& nn1 = GetParam().createNN();
//		const std::string filename = "../resources/to_file_test.json";
//		save_to_file(filename, nn1);
//		const NeuralNetwork nn2 = load_from_file(filename);
//		ASSERT_EQ(nn1, nn2);
//	}
//
//	TEST_P(NNForwardFixture, CorrectlyPassesForward)
//	{
//		const auto& param = GetParam();
//		const NeuralNetwork nn = load_from_file(param.filename);
//		AssertNear(nn(param.in), param.out, param.abs_error);
//	}
//}
