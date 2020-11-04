#pragma once

#include <vector>

namespace openn
{
	struct Node
	{
		explicit Node(size_t prev_layer_size);

		double val;
		std::vector<double> w;
		double b;
	};

	using Layer = std::vector<Node>;

	class NeuralNetwork
	{
	public:
		NeuralNetwork(size_t input_size, size_t output_size);
		virtual ~NeuralNetwork() = default;

		virtual void addLayer(size_t size);

	protected:
		std::vector<Layer> layers;
	};
}