#pragma once

#include <functional>

namespace openn
{
	using float_t = double;

	struct Node;
	struct Layer;
	struct NeuralNetwork;

	struct LayerStructure;
	
	using ActivationF = std::function<float_t(float_t)>;
	enum class ActivationFType { sigmoid, ReLU, softplus, tanh, _SIZE };
}
