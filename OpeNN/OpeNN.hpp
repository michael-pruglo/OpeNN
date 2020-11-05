#pragma once

#include <ostream>
#include <vector>

namespace openn
{
	using float_t = double;

	struct Node
	{
		explicit Node(size_t prev_layer_size = 0);

		float_t val;
		std::vector<float_t> w;
		float_t b;
	};

	using Layer = std::vector<Node>;

	struct NeuralNetwork
	{
		/// construct a network with 1 input and 1 output layer
		NeuralNetwork(size_t input_size, size_t output_size);
		virtual ~NeuralNetwork() = default;

		virtual void addLayer(size_t layer_size);
		virtual void addLayer(size_t layer_size, size_t pos);

		std::vector<Layer> layers;
	};

	std::ostream& operator<<(std::ostream & os, const NeuralNetwork& nn);
}