#pragma once

#include <core/types.hpp>
#include <functional>

namespace openn
{
    using core::float_t;
    using core::Vec;
    using core::Matrix;

    using AlgebraicF = std::function<float_t(float_t)>;
    enum class ActivationFType { SIGMOID, ReLU, SOFTPLUS, TANH };
    float_t activation_f(ActivationFType type, float_t x);
    Vec     activation_f(ActivationFType type, const Vec& v);
    float_t derivative_f(ActivationFType type, float_t x);
    Vec     derivative_f(ActivationFType type, const Vec& v);

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
