#include <openn/openn.hpp>
#include <core/random.hpp>
#include <core/algebra.hpp>

using namespace openn;
using core::operator*;
using core::operator+;

bool openn::operator==(const LayerMetadata& lm1, const LayerMetadata& lm2)
{
    return lm1.size == lm2.size && lm1.activation == lm2.activation;
}

FeedForwardNetwork::FeedForwardNetwork(const std::vector<LayerMetadata>& nn_metadata)
{
    layers = core::generate_i(nn_metadata.size()-1, [&nn_metadata](size_t i){
        return Layer(nn_metadata[i+1], nn_metadata[i].size);
    });
}

LayerMetadata FeedForwardNetwork::get_layer_metadata(size_t i) const
{
    return layers.at(i).metadata;
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
FeedForwardNetwork::Layer::Layer(LayerMetadata metadata, size_t prev_layer_size)
    : w(core::rand_matrix(metadata.size, prev_layer_size))
    , bias(core::rand_vec(metadata.size))
    , metadata(metadata)
{
}

Vec FeedForwardNetwork::Layer::activation_f(const Vec& v) const
{
    return core::map(ACTIVATION_FUNCTIONS.at(metadata.activation), v);
}
Vec FeedForwardNetwork::Layer::derivative_f(const Vec& v) const
{
    return core::map(DERIVATIVE_FUNCTIONS.at(metadata.activation), v);
}