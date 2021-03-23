#pragma once

#include <openn/types.hpp>
#include <nlohmann/json.hpp>

namespace openn
{
    class FeedForwardNetwork : public INeuralNetwork
    {
    public:
        struct LayerInitRandData{ size_t size=0; ActivationFType activation_type=ActivationFType::sigmoid; };
        FeedForwardNetwork(size_t input_size, const std::vector<LayerInitRandData>& nn_structure);

        struct LayerInitValuesData{ ActivationFType activation_type=ActivationFType::sigmoid; WnB wnb; };
        FeedForwardNetwork(const std::vector<LayerInitValuesData>& values);

        ~FeedForwardNetwork() override = default;

        Vec operator()(const Vec& input) const override;

    protected:
        struct Layer;
        std::vector<Layer> layers;
    };

    struct FeedForwardNetwork::Layer : public WnB
    {
        Layer(size_t prev_size, size_t size, ActivationFType activation_type);
        Layer(ActivationFType activation_type, WnB wnb);

        Vec activation_f(const Vec& v) const;
        Vec derivative_f(const Vec& v) const;

        ActivationFType activation_type;
    };
}
