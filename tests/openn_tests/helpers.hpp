#pragma once

#include <tests/common/helpers.hpp>
#include <openn/types.hpp>
#include <openn/openn.hpp>

namespace openn
{
    ActivationFType rand_activation();

    class TestableNeuralNetwork : public NeuralNetwork
    {
    public:
        using NeuralNetwork::NeuralNetwork;
        void set_layer(size_t idx, Matrix w, Vec bias);
    };
}