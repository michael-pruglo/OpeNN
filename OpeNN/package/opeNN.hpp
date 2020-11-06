#pragma once

#include <ostream>
#include <vector>

namespace openn
{
	using float_t = double;

	struct Node
	{
		explicit Node(size_t prev_layer_size = 0);
		void resetWeight(size_t prev_layer_size = 0);

		std::vector<float_t> w;
		float_t bias;
	};

	struct Layer : public std::vector<Node>
	{
		explicit Layer(size_t layer_size, size_t prev_layer_size = 0);
		void resetWeights(size_t prev_layer_size = 0);
	};

	struct NeuralNetwork
	{
		/// construct a network with 1 input and 1 output layer
		NeuralNetwork(size_t input_size = 0, size_t output_size = 0);
		virtual ~NeuralNetwork() = default;

		virtual					void addLayer(size_t layer_size = 0);
		virtual					void addLayer(size_t layer_size, size_t pos);

		virtual std::vector<float_t> operator()(const std::vector<float_t>& input);

	private:
				std::vector<float_t> _forward(const std::vector<float_t>& prev, size_t idx);
		static				 float_t _calcVal(const Node& node, const std::vector<float_t>& prev);
		static inline		 float_t activationF(float_t x);
		
	public:
		std::vector<Layer> layers;
	};

	struct ActivationF
	{
		static float_t ReLU		(float_t x) { return std::max(0., x); } 
		static float_t sigmoid	(float_t x) { return 1. / (1. + std::exp(-x)); }
		static float_t softplus	(float_t x) { return std::log(1. + std::exp(x)); }
		static float_t tanh		(float_t x) { return std::tanh(x); }
	};
}