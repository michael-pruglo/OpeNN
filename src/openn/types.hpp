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

    class INeuralNetwork
    {
    public:
        virtual ~INeuralNetwork() = default;
        virtual Vec operator()(const Vec& input) const = 0;
    };

    struct WnB
    {
        Matrix w;
        Vec bias;
    };
    using Gradient = WnB;
}
