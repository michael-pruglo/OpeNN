#pragma once

#include <openn/types.hpp>
#include <nlohmann/json.hpp>

namespace openn
{
    class NeuralNetwork : public INeuralNetwork
    {
    public:
        explicit NeuralNetwork(const std::vector<LayerMetadata>& nn_metadata = { {}, {} });
        ~NeuralNetwork() override = default;

        LayerMetadata get_layer_metadata(size_t i) const override;
        Vec operator()(const Vec& input) const override;

    private:
        friend class NeuralNetworkPrinter;
        friend void to_json(nlohmann::json& j, const NeuralNetwork& nn);
        friend void from_json(const nlohmann::json& j, NeuralNetwork& nn);

    protected:
        struct Layer;
        std::vector<Layer> layers;
    };

    struct NeuralNetwork::Layer
    {
        explicit Layer(LayerMetadata metadata = {}, size_t prev_layer_size = 0);

        Vec activation_f(const Vec& v) const;
        Vec derivative_f(const Vec& v) const;

        inline size_t size() const { return metadata.size; }

        Matrix w;
        Vec bias;
        LayerMetadata metadata;
    };
}
