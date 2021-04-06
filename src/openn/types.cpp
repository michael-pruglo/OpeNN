#include <src/openn/types.hpp>
#include <core/algebra.hpp>
#include <core/utility.hpp>

using namespace openn;

namespace
{
    using openn::float_t;
    using AlgebraicF = std::function<Vec(const Vec&)>;

    const std::unordered_map<ActivationFType, AlgebraicF> ACTIVATION_FUNCTIONS = {
        { ActivationFType::SIGMOID,  core::sigmoid },
        { ActivationFType::ReLU,     core::relu },
        { ActivationFType::SOFTPLUS, core::softplus },
        { ActivationFType::TANH,     core::tanh },
    };

    const std::unordered_map<ActivationFType, AlgebraicF> DERIVATIVE_FUNCTIONS = {
        { ActivationFType::SIGMOID,  core::der_sigmoid },
        { ActivationFType::ReLU,     core::der_relu },
        { ActivationFType::SOFTPLUS, core::der_softplus },
        { ActivationFType::TANH,     core::der_tanh },
    };
}

Vec openn::activation_f(ActivationFType type, const Vec& v)
{
    const auto& f = ACTIVATION_FUNCTIONS.at(type);
    return f(v);
}

Vec openn::derivative_f(ActivationFType type, const Vec& v)
{
    const auto& f = DERIVATIVE_FUNCTIONS.at(type);
    return f(v);
}

void openn::operator+=(Gradient& grad, const Gradient& addend)
{
    grad.w = grad.w + addend.w;
    grad.bias = grad.bias + addend.bias;
}

void openn::operator/=(Gradient& grad, float_t divisor)
{
    grad.w = grad.w / divisor;
    grad.bias = grad.bias / divisor;
}
