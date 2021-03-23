#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>
#include <openn/types.hpp>

namespace openn::types
{
    void expect_layer_structure(auto layer, size_t prev_size, size_t curr_size, ActivationFType activ_type)
    {
        EXPECT_EQ(layer.bias.size(), curr_size);
        EXPECT_EQ(layer.w.rows(), curr_size);
        EXPECT_EQ(layer.w.cols(), prev_size);
        EXPECT_EQ(layer.activation_type, activ_type);
    }

    TEST(OpennTypesTest, InitNNRand)
    {
        TransparentFFN tffn1({ 3,
        {
            { 1, ActivationFType::sigmoid },
            { 5, ActivationFType::ReLU },
            { 4, ActivationFType::softplus },
            { 2, ActivationFType::tanh },
        }});
        const auto& lrs1 = tffn1.get_layers();
        expect_layer_structure(lrs1[0], 3, 1, ActivationFType::sigmoid);
        expect_layer_structure(lrs1[1], 1, 5, ActivationFType::ReLU);
        expect_layer_structure(lrs1[2], 5, 4, ActivationFType::softplus);
        expect_layer_structure(lrs1[3], 4, 2, ActivationFType::tanh);
    }

    TEST(OpennTypesTest, InitNNValues)
    {

    }
}