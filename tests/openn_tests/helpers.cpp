#include <tests/openn_tests/helpers.hpp>
#include <core/random.hpp>

namespace openn
{
    ActivationFType rand_activation()
    {
        const auto sz = static_cast<size_t>(ActivationFType::_SIZE);
        const auto idx = core::rand_i<size_t>(0, sz-1U);
        return static_cast<const ActivationFType>(idx);
    }

    void TestableNeuralNetwork::set_layer(size_t idx, Matrix w, Vec bias)
    {
        auto& layer = layers[idx];

        assert(w.rows() == layer.metadata.size);
        assert(bias.size() == layer.metadata.size);

        layer.w = std::move(w);
        layer.bias = std::move(bias);
    }
}
