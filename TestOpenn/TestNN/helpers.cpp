#include <TestOpenn/TestNN/helpers.hpp>
#include <OpeNN/package/io/nn_printer.cpp> //for NeuralNetwork::operator<<()

namespace openn
{
	void testNode(const Node& n, size_t inputs_count)
	{
		ASSERT_EQ(n.w.size(), inputs_count);
	}

	void testLayer(const Layer& l, size_t prev_layer_size)
	{
		for (const auto& n : l)
			testNode(n, prev_layer_size);
	}

	void testNet(const NeuralNetwork& nn, const std::vector<LayerMetadata>& layer_structure)
	{
		const size_t N = layer_structure.size();
		ASSERT_EQ(nn.layers.size(), N);

		for (size_t i = 0; i < N; ++i)
		{
			ASSERT_EQ(nn.layers[i].size(), layer_structure[i].size) 
				<< nn << "exp size ["<<i<<"]: " << layer_structure[i].size;
			ASSERT_EQ(nn.layers[i].activation, layer_structure[i].activation)
				<< nn << "exp act ["<<i<<"]: " << to_string(layer_structure[i].activation);
			testLayer(nn.layers[i], i ? layer_structure[i-1].size : 0);
		}
	}

	ActivationFType rand_activation()
	{
		const size_t idx = randi(0, static_cast<int>(ActivationFType::_SIZE)-1);
		return static_cast<ActivationFType>(idx);
	}
}
