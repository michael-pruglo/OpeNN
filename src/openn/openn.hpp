#pragma once

#include <openn/types.hpp>

namespace openn
{
    class FeedForwardNetwork : public NeuralNetwork
    {
    public:
        //construct randomized
        FeedForwardNetwork(const std::vector<size_t>& layer_sizes, std::vector<ActivationFType> activation_types);
        FeedForwardNetwork(const std::vector<size_t>& layer_sizes, ActivationFType universal_activation = ActivationFType::SIGMOID);

        //construct with given weights and biases
        FeedForwardNetwork(Matrixes weights, Vectors biases, std::vector<ActivationFType> activation_types);
        FeedForwardNetwork(Matrixes weights, Vectors biases, ActivationFType universal_activation = ActivationFType::SIGMOID);

        ~FeedForwardNetwork() override = default;

    public:
        Vec forward(const Vec& input) override;
        Gradient backprop(const Vec& expected, CostFType cost_f_type) override;
        void update(const Gradient& grad, float_t eta) override;

    protected:
        size_t layers_count; //includes the input layer
        //matrixes and vectors are indexed as follows:
        //[0] - input layer
        //[1] - first hidden layer
        //[2] - second hidden layer
        //...
        //[layers_count-1] - output layer
        Matrixes w;
        Vectors b, z, a;
        std::vector<ActivationFType> activation_types;
    };
}
