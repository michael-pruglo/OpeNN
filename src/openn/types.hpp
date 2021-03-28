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

    struct WnB
    {
        void operator+=(const WnB& other);
        void operator/=(float_t x);

        Matrix w;
        Vec bias;
    };
    using Gradient = WnB;

    class NeuralNetwork
    {
    public:
        virtual ~NeuralNetwork() = default;
        virtual Vec operator()(const Vec& input) const = 0;
        virtual void update(const Gradient& grad, float_t eta) = 0;
    };
}
