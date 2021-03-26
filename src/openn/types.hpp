#pragma once

#include <core/types.hpp>
#include <functional>

namespace openn
{
    using core::float_t;
    using core::Vec;
    using core::Matrix;

    enum class ActivationFType { SIGMOID, ReLU, SOFTPLUS, TANH };
    float_t activation_f(ActivationFType type, float_t x);
    Vec     activation_f(ActivationFType type, const Vec& v);
    float_t derivative_f(ActivationFType type, float_t x);
    Vec     derivative_f(ActivationFType type, const Vec& v);

    // cost/loss/objective function
    enum class CostFType { MSE, CROSS_ENTROPY };
    float_t cost_f(CostFType type, const Vec& v, const Vec& exp);

    class NeuralNetwork
    {
    public:
        virtual ~NeuralNetwork() = default;
        virtual Vec operator()(const Vec& input) const = 0;
    };

    struct WnB
    {
        Matrix w;
        Vec bias;
    };
    using Gradient = WnB;
}
