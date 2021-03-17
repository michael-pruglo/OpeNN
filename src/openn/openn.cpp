#include <openn/openn.hpp>
#include <core/random.hpp>
#include <core/algebra.hpp>

using namespace openn;

bool openn::operator==(const LayerMetadata& lm1, const LayerMetadata& lm2)
{
    return lm1.size == lm2.size && lm1.activation == lm2.activation;
}

NeuralNetwork::NeuralNetwork(const std::vector<LayerMetadata>& nn_structure)
{
    const size_t n = nn_structure.size();
    layers.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        const auto& prev_size = i ? nn_structure[i-1].size : 0;
        layers.emplace_back(nn_structure[i].size, prev_size, nn_structure[i].activation);
    }
}

LayerMetadata NeuralNetwork::getLayerMetadata(size_t i) const
{
    return { layers[i].size(), layers[i].activation };
}

Vec NeuralNetwork::operator()(const Vec& input) const
{
    return _forward(input, 1);
}

Vec NeuralNetwork::_forward(const Vec& input, size_t idx) const
{
    if (idx == layers.size())
        return input;

    using core::operator*;
    using core::operator+;
    const auto output = layers[idx].activation_f(layers[idx].w * input + layers[idx].bias);

    return _forward(output, idx+1);
}

bool NeuralNetwork::operator==(const NeuralNetwork& other) const
{
    return layers == other.layers;
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
            { ActivationFType::ReLU,        [](float_t x) -> float_t { return x > 0; } },
            { ActivationFType::sigmoid,     [](float_t x) -> float_t { const auto& f = ACTIVATION_FUNCTIONS.at(ActivationFType::sigmoid); return f(x)*(1. - f(x)); } },
            { ActivationFType::softplus,    [](float_t x) -> float_t { return 1. / (1. + std::exp(-x)); } },
            { ActivationFType::tanh,        [](float_t x) -> float_t { return 1. - std::pow(std::tanh(x), 2); } },
    };

}
NeuralNetwork::Layer::Layer(size_t layer_size, size_t prev_layer_size, ActivationFType activation_)
    : w(core::rand_matrix(layer_size, prev_layer_size))
    , bias(core::rand_vec(layer_size))
    , activation(activation_)
    , _act_f(ACTIVATION_FUNCTIONS.at(activation_))
    , _der_f(DERIVATIVE_FUNCTIONS.at(activation_))
{
}

Vec NeuralNetwork::Layer::activation_f(const Vec& v) const
{
    return core::map(_act_f, v);
}
Vec NeuralNetwork::Layer::derivative_f(const Vec& v) const
{
    return core::map(_der_f, v);
}

bool NeuralNetwork::Layer::operator==(const Layer& other) const
{
    return w == other.w && bias == other.bias && activation == other.activation;
}
