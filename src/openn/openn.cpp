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

void FeedForwardNetwork::update(const Gradient& grad, float_t eta)
{
    throw "not implemented";
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