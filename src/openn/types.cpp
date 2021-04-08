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

Vec openn::activation_der(ActivationFType type, const Vec& v)
{
    const auto& f = DERIVATIVE_FUNCTIONS.at(type);
    return f(v);
}


namespace
{
    using CostF = std::function<openn::float_t(const Vec&,const Vec&)>;

    const std::unordered_map<CostFType, CostF> COST_FUNCTIONS = {
        { CostFType::MEAN_SQUARED_ERROR, core::mean_squared_eror },
        { CostFType::CROSS_ENTROPY,      core::cross_entropy },
    };

    const std::unordered_map<CostFType, CostF> COST_DERIVATIVES = {
        { CostFType::MEAN_SQUARED_ERROR, core::der_mean_squared_eror },
        { CostFType::CROSS_ENTROPY,      core::der_cross_entropy },
    };
}

core::float_t openn::cost_f(CostFType type, const Vec &v, const Vec &exp)
{
    const auto& f = COST_FUNCTIONS.at(type);
    return f(v, exp);
}

core::float_t openn::cost_der(CostFType type, const Vec &v, const Vec &exp)
{
    const auto& f = COST_DERIVATIVES.at(type);
    return f(v, exp);
}
