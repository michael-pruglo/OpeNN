#pragma once

#include <core/types.hpp>
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
        size_t size = 0;
        ActivationFType activation = ActivationFType::sigmoid;
    };
    bool operator==(const LayerMetadata& lm1, const LayerMetadata& lm2);

    class INeuralNetwork
    {
    public:
        virtual ~INeuralNetwork() = default;

        virtual LayerMetadata   get_layer_metadata(size_t i) const = 0;
        virtual Vec	            operator()(const Vec& input) const = 0;
    };
}
