#include <openn/openn.hpp>
#include <core/random.hpp>

using namespace openn;
using xt::linalg::dot;


FeedForwardNetwork::FeedForwardNetwork(
    const std::vector<size_t>& layer_sizes,
    std::vector<ActivationFType> activation_functions
)
    : FeedForwardNetwork(
        [&layer_sizes](){
            const size_t L = layer_sizes.size();
            Matrixes weights({L});

            for (size_t i = 1; i < L; ++i)
                weights[i-1] = core::rand_tensor<Matrix>({layer_sizes[i], layer_sizes[i-1]});

            return weights;
        }(),
        [&layer_sizes](){
            const size_t L = layer_sizes.size();
            Vectors biases({L});

            for (size_t i = 1; i < L; ++i)
                biases[i-1] = core::rand_tensor<Vec>({layer_sizes[i]});

            return biases;
        }(),
        std::move(activation_functions)
    )
{
}

FeedForwardNetwork::FeedForwardNetwork(const std::vector<size_t>& layer_sizes, ActivationFType universal_activation)
    : FeedForwardNetwork(
        layer_sizes,
        std::vector<ActivationFType>(layer_sizes.size(), universal_activation)
    )
{
}

FeedForwardNetwork::FeedForwardNetwork(Matrixes weights, Vectors biases, std::vector<ActivationFType> activation_types)
    : layers_count(activation_types.size()+1)
    , w(std::move(weights))
    , b(std::move(biases))
    , z({layers_count})
    , a({layers_count})
    , activation_types(std::move(activation_types))
{
    assert(weights.size() == layers_count);
    assert(biases.size()  == layers_count);

    for (size_t i = 0; i < layers_count; ++i)
    {
        z[i] = xt::zeros_like(b[i]);
        a[i] = xt::zeros_like(b[i]);
    }
}

FeedForwardNetwork::FeedForwardNetwork(Matrixes weights, Vectors biases, ActivationFType universal_activation)
    : FeedForwardNetwork(
        std::move(weights),
        std::move(biases),
        std::vector<ActivationFType>(weights.size(), universal_activation)
    )
{
    assert(weights.size() == biases.size());
}


Vec FeedForwardNetwork::forward(const Vec& input)
{
    a[0] = input;
    for (int i = 1; i < layers_count; ++i)
    {
        z[i] = dot(w[i], a[i-1]) + b[i];
        a[i] = activation_f(activation_types[i], z[i]);
    }
    return a[-1];
}

Gradient FeedForwardNetwork::backprop(const Vec& expected, CostFType cost_f_type)
{
    Gradient grad{ .w = Matrixes({layers_count}), .b = Vectors({layers_count}) };
    for (size_t i = 0; i < layers_count; ++i)
    {
        grad.w[i] = xt::zeros_like(w[i]);
        grad.b[i] = xt::zeros_like(b[i]);
    }


    //last layer
    Vec delta = cost_der(cost_f_type, a[-1], expected)
        * activation_der(activation_types.back(), z[-1]);
    grad.w[-1] = dot(delta, xt::transpose(a[-2]));
    grad.b[-1] = delta;

    //prev layers
    for (int l = 2; l < layers_count; ++l)
    {
        const auto& trans = xt::transpose(w[-l+1]);
        const Vec dotpr = dot(trans, delta);
        const Vec sp = activation_der(activation_types[-l], z[-l]);
        const auto& res = dotpr * sp;
        const Vec vec_res = res;
        delta = vec_res;
        grad.w[-l] = dot(delta, xt::transpose(a[-l-1]));
        grad.b[-l] = delta;
    }

    return grad;
}

void FeedForwardNetwork::update(const Gradient& grad, float_t eta)
{
    w += -eta * grad.w;
    b += -eta * grad.b;
}

