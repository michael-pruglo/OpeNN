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
        res = layer.activation_f(layer.w * res + layer.bias);
    return res;
}



namespace
{
    using openn::float_t;

    const std::unordered_map<ActivationFType, AlgebraicF> ACTIVATION_FUNCTIONS = {
        { ActivationFType::ReLU,        [](float_t x) -> float_t { return std::max(0., x); } },
        { ActivationFType::sigmoid,     [](float_t x) -> float_t { return 1. / (1. + std::exp(-x)); } },
        { ActivationFType::softplus,    [](float_t x) -> float_t { return std::log(1. + std::exp(x)); } },
        { ActivationFType::tanh,        [](float_t x) -> float_t { return std::tanh(x); } },
    };

    const std::unordered_map<ActivationFType, AlgebraicF> DERIVATIVE_FUNCTIONS = {
        { ActivationFType::ReLU,        [](float_t x) -> float_t { return x > 0.; } },
        { ActivationFType::sigmoid,     [](float_t x) -> float_t { const auto& f = ACTIVATION_FUNCTIONS.at(ActivationFType::sigmoid); return f(x)*(1. - f(x)); } },
        { ActivationFType::softplus,    [](float_t x) -> float_t { return 1. / (1. + std::exp(-x)); } },
        { ActivationFType::tanh,        [](float_t x) -> float_t { return 1. - std::pow(std::tanh(x), 2); } },
    };

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

Vec FeedForwardNetwork::Layer::activation_f(const Vec& v) const
{
    return core::map(ACTIVATION_FUNCTIONS.at(activation_type), v);
}
Vec FeedForwardNetwork::Layer::derivative_f(const Vec& v) const
{
    return core::map(DERIVATIVE_FUNCTIONS.at(activation_type), v);
}
