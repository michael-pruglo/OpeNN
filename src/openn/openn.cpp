#include <openn/openn.hpp>
#include <core/random.hpp>

using namespace openn;
using xt::linalg::dot;

FeedForwardNetwork::FeedForwardNetwork(size_t input_size, const std::vector<LayerInitRandData>& nn_structure)
{
    const size_t n = nn_structure.size();
    layers.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        layers.emplace_back(
            i ? nn_structure[i-1].size : input_size,
            nn_structure[i].size,
            nn_structure[i].activation_type
        );
    }
}

FeedForwardNetwork::FeedForwardNetwork(const std::vector<LayerInitValuesData>& values)
{
    const size_t n = values.size();
    layers.reserve(n);
    for (size_t i = 0; i < n; ++i)
        layers.emplace_back(values[i].activation_type, values[i].wnb);
}

Vec FeedForwardNetwork::operator()(const Vec& input) const
{
    auto a = input;
    for (const auto& layer: layers)
    {
        const auto z = dot(layer.w, a) + layer.bias;
        a = activation_f(layer.activation_type, z);
    }
    return a;
}

void FeedForwardNetwork::update(const Gradient& grad, float_t eta)
{
    throw "not implemented";
}


FeedForwardNetwork::Layer::Layer(size_t prev_size, size_t size, ActivationFType activation_type)
    : WnB{
        .w = core::rand_tensor<Matrix>({size, prev_size}),
        .bias = core::rand_tensor<Vec>({size})
    }
    , activation_type(activation_type)
{
}

FeedForwardNetwork::Layer::Layer(ActivationFType activation_type, WnB wnb)
    : WnB(std::move(wnb))
    , activation_type(activation_type)
{
}