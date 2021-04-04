#pragma once

#include <openn/types.hpp>

namespace openn
{
    class FeedForwardNetwork : public NeuralNetwork
    {
    public:
        struct LayerInitRandData{ size_t size=0; ActivationFType activation_type=ActivationFType::SIGMOID; };
        FeedForwardNetwork(size_t input_size, const std::vector<LayerInitRandData>& nn_structure);

        struct LayerInitValuesData{ ActivationFType activation_type=ActivationFType::SIGMOID; WnB wnb; };
        FeedForwardNetwork(const std::vector<LayerInitValuesData>& values);

        ~FeedForwardNetwork() override = default;

    public:
        Vec operator()(const Vec& input) const override;
        virtual void update(const Gradient& grad, float_t eta) override;

    protected:
        struct Layer;
        std::vector<Layer> layers;
    };

    struct FeedForwardNetwork::Layer : public WnB
    {
        Layer(size_t prev_size, size_t size, ActivationFType activation_type);
        Layer(ActivationFType activation_type, WnB wnb);

        ActivationFType activation_type;
    };
}
