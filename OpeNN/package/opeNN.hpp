#pragma once

#include <OpeNN/package/types.hpp>
#include <unordered_map>
#include <vector>

namespace openn
{
	struct Node
	{
		explicit Node(size_t prev_layer_size = 0);
		
		void resetWeight(size_t prev_layer_size = 0);

		std::vector<float_t> w;
		float_t bias;
	};

	struct LayerStructure
	{
		LayerStructure(size_t size, ActivationFType activation = ActivationFType::sigmoid);

		size_t size;
		ActivationFType activation;
	};

	struct Layer : public std::vector<Node>
	{
		explicit Layer(size_t layer_size = 0, size_t prev_layer_size = 0, ActivationFType activation = ActivationFType::sigmoid);
		
		float_t activation_f(float_t x) const;

		void resetWeights(size_t prev_layer_size = 0);
		LayerStructure getLayerStructure() const;

		ActivationFType activation;
		static std::unordered_map<ActivationFType, ActivationF> activation_functions;
	};

	struct NeuralNetwork
	{
		explicit NeuralNetwork(const std::vector<LayerStructure>& nn_structure = {0, 0});
		virtual ~NeuralNetwork() = default;
		
		virtual					void addLayer(size_t layer_size = 0, ActivationFType activation = ActivationFType::sigmoid);
		virtual					void addLayer(size_t layer_size, size_t pos, ActivationFType activation = ActivationFType::sigmoid);
		 std::vector<LayerStructure> getNNStructure() const;

		virtual std::vector<float_t> operator()(const std::vector<float_t>& input) const;

	private:
				std::vector<float_t> _forward(const std::vector<float_t>& prev, size_t idx) const;
		static				 float_t _calcVal(const Node& node, const std::vector<float_t>& prev);
		
	public:
		std::vector<Layer> layers;
	};

	/// Implementation details
	bool operator==(const Node& n1, const Node& n2);
	bool operator==(const NeuralNetwork& nn1, const NeuralNetwork& nn2);
}