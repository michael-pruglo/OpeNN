#pragma once

#include <core/types.hpp>
#include <xtensor-blas/xlinalg.hpp>

namespace openn
{
    using core::float_t;
    using core::Vec;
    using core::Vectors;
    using core::Matrix;
    using core::Matrixes;


    enum class ActivationFType
    {
        SIGMOID, //a.k.a. logistic
        ReLU,
        SOFTPLUS,
        TANH,
    };
    Vec activation_f  (ActivationFType type, const Vec& v);
    Vec activation_der(ActivationFType type, const Vec& v);


    // cost/loss/objective function
    enum class CostFType
    {
        MEAN_SQUARED_ERROR,
        CROSS_ENTROPY,
    };
    Vec cost_f  (CostFType type, const Vec& v, const Vec& exp);
    Vec cost_der(CostFType type, const Vec& v, const Vec& exp);


    struct Gradient
    {
        Matrixes w;
        Vectors  b;
    };


    class NeuralNetwork
    {
    public:
        virtual ~NeuralNetwork() = default;

        virtual Vec      forward (const Vec& input) = 0;
        virtual Gradient backprop(const Vec& expected, CostFType cost_f_type) = 0;
        virtual void     update  (const Gradient& grad, float_t eta) = 0;
    };
}
