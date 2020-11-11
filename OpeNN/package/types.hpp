#pragma once

#include <Core/types.hpp>
#include <functional>

namespace openn
{
	using core::float_t;
	using core::Vec;
	using core::Matrix;

	using AlgebraicF = std::function<float_t(float_t)>;
	enum class ActivationFType { sigmoid, ReLU, softplus, tanh, _SIZE };

	struct LayerMetadata
	{
		LayerMetadata(size_t size = 0, ActivationFType activation = ActivationFType::sigmoid);

		size_t size;
		ActivationFType activation;
	};

	class INeuralNetwork
	{
	public:
		virtual ~INeuralNetwork() = default;
		
		virtual LayerMetadata	getLayerMetadata(size_t i) const = 0;
		virtual Vec				operator()(const Vec& input) const = 0;
	};
}
