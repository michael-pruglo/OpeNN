#include <tests/openn_tests/helpers.hpp>
#include <openn/openn.hpp>

namespace openn
{
    //at this point we have to trust getLayerMetadata
    void test_nn_structure(const INeuralNetwork& nn, const std::vector<LayerMetadata>& nn_structure)
    {
        using openn::operator==;
        for (size_t i = 0; i < nn_structure.size(); ++i)
            ASSERT_EQ(nn.getLayerMetadata(i), nn_structure[i]);
    }

    ActivationFType rand_activation()
    {
        return ActivationFType::sigmoid;
//        const auto sz = static_cast<size_t>(ActivationFType::_SIZE);
//        const auto idx = core::rand_i<size_t>(0, sz-1u);
//        return static_cast<ActivationFType>(idx);
    }
}
