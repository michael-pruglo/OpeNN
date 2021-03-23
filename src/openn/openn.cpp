#include <openn/openn.hpp>
#include <core/random.hpp>
#include <core/algebra.hpp>

using namespace openn;
using core::operator*;
using core::operator+;

FeedForwardNetwork::FeedForwardNetwork(size_t input_size, const std::vector<LayerInitRandData>& nn_structure)
{
    layers = core::generate_i(nn_structure.size(), [&nn_structure, input_size](size_t i){
        return Layer(
            i ? nn_structure[i-1].size : input_size,
            nn_structure[i].size,
            nn_structure[i].activation_type
        );
    });
}

FeedForwardNetwork::FeedForwardNetwork(const std::vector<LayerInitValuesData>& values)
{
    layers = core::generate_i(values.size(), [&values](size_t i){
        return Layer(values[i].activation_type, values[i].wnb);
    });
}

Vec FeedForwardNetwork::operator()(const Vec& input) const
{
    auto res = input;
    for (const auto& layer: layers)
        res = activation_f(layer.activation_type, layer.w * res + layer.bias);
    return res;
}

FeedForwardNetwork::Layer::Layer(size_t prev_size, size_t size, ActivationFType activation_type)
    : WnB{
    .w = core::rand_matrix(size, prev_size),
    .bias = core::rand_vec(size)
}
    , activation_type(activation_type)
{
}

FeedForwardNetwork::Layer::Layer(ActivationFType activation_type, WnB wnb)
    : WnB(std::move(wnb))
    , activation_type(activation_type)
{
}


namespace
{
    using openn::float_t;

    const std::unordered_map<ActivationFType, AlgebraicF> ACTIVATION_FUNCTIONS = {
        { ActivationFType::ReLU,     [](float_t x) -> float_t { return std::max(0., x); } },
        { ActivationFType::SIGMOID,  [](float_t x) -> float_t { return 1. / (1. + std::exp(-x)); } },
        { ActivationFType::SOFTPLUS, [](float_t x) -> float_t { return std::log(1. + std::exp(x)); } },
        { ActivationFType::TANH,     [](float_t x) -> float_t { return std::tanh(x); } },
    };

    const std::unordered_map<ActivationFType, AlgebraicF> DERIVATIVE_FUNCTIONS = {
        { ActivationFType::ReLU,     [](float_t x) -> float_t { return x > 0.; } },
        { ActivationFType::SIGMOID,  [](float_t x) -> float_t { const auto& f = ACTIVATION_FUNCTIONS.at(ActivationFType::SIGMOID); return f(x)*(1. - f(x)); } },
        { ActivationFType::SOFTPLUS, [](float_t x) -> float_t { return 1. / (1. + std::exp(-x)); } },
        { ActivationFType::TANH,     [](float_t x) -> float_t { return 1. - std::pow(std::tanh(x), 2); } },
    };

}

core::float_t openn::activation_f(ActivationFType activation_type, float_t x)
{
    const auto& f = ACTIVATION_FUNCTIONS.at(activation_type);
    return f(x);
}
Vec openn::activation_f(ActivationFType activation_type, const Vec& v)
{
    const auto& f = ACTIVATION_FUNCTIONS.at(activation_type);
    return core::map(f, v);
}

core::float_t openn::derivative_f(ActivationFType activation_type, float_t x)
{
    const auto& f = DERIVATIVE_FUNCTIONS.at(activation_type);
    return f(x);
}
Vec openn::derivative_f(ActivationFType activation_type, const Vec& v)
{
    const auto& f = DERIVATIVE_FUNCTIONS.at(activation_type);
    return core::map(f, v);
}
