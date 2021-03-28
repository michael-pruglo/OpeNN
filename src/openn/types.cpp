#include <src/openn/types.hpp>
#include <core/algebra.hpp>
#include <core/utility.hpp>

using namespace openn;
using core::operator+;
using core::operator/;
using core::operator*;

namespace
{
    using openn::float_t;
    using AlgebraicF = std::function<float_t(float_t)>;

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

core::float_t openn::activation_f(ActivationFType type, float_t x)
{
    const auto& f = ACTIVATION_FUNCTIONS.at(type);
    return f(x);
}
Vec openn::activation_f(ActivationFType type, const Vec& v)
{
    const auto& f = ACTIVATION_FUNCTIONS.at(type);
    return core::map(f, v);
}

core::float_t openn::derivative_f(ActivationFType type, float_t x)
{
    const auto& f = DERIVATIVE_FUNCTIONS.at(type);
    return f(x);
}
Vec openn::derivative_f(ActivationFType type, const Vec& v)
{
    const auto& f = DERIVATIVE_FUNCTIONS.at(type);
    return core::map(f, v);
}

void WnB::operator+=(const WnB& other)
{
    w = w + other.w;
    bias = bias + other.bias;
}

void WnB::operator/=(core::float_t x)
{
    w = w / x;
    bias = bias / x;
}
