#pragma once

#include <OpeNN/package/types.hpp>
#include <packages/nlohmann/json.hpp>

namespace openn
{
	class NeuralNetwork : public INeuralNetwork
	{
	public:
		explicit NeuralNetwork(const std::vector<LayerMetadata>& nn_structure = {0, 0});
		
		inline LayerMetadata getLayerMetadata(size_t i) const override;
		inline Vec operator()(const Vec& input) const override;
		inline bool operator==(const NeuralNetwork& other) const;

	private:
		Vec _forward(const Vec& input, size_t idx) const;
	private:
		friend class NeuralNetworkPrinter;
		friend void to_json(nlohmann::json& j, const NeuralNetwork& nn);
		friend void from_json(const nlohmann::json& j, NeuralNetwork& nn);

	private:
		struct Layer;
		std::vector<Layer> layers;
	};

	struct NeuralNetwork::Layer
	{
		explicit Layer(
			size_t layer_size = 0, 
			size_t prev_layer_size = 0, 
			ActivationFType activation = ActivationFType::sigmoid
		);
		
		Vec activation_f(const Vec& v) const;
		Vec derivative_f(const Vec& v) const;

		inline size_t size() const { return bias.size(); }
		inline bool operator==(const Layer& other) const;

		Matrix w; 
		Vec bias;
		ActivationFType activation;
		AlgebraicF _act_f, _der_f;
	};
}
