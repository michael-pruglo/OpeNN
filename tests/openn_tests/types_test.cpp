#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>

namespace openn::types
{
    void expect_layer_structure(auto layer, size_t prev_size, size_t curr_size, ActivationFType activ_type)
    {
        EXPECT_EQ(layer.bias.size(), curr_size);
        EXPECT_EQ(layer.w.rows(), curr_size);
        EXPECT_EQ(layer.w.cols(), prev_size);
        EXPECT_EQ(layer.activation_type, activ_type);
    }

    TEST(OpennTypesTest, InitNNRandDef)
    {
        TransparentFFN tffn_def(7, {});
        const auto& lrs_def = tffn_def.get_layers();
        ASSERT_EQ(lrs_def.size(), 0);
    }

    TEST(OpennTypesTest, InitNNRand1)
    {
        TransparentFFN tffn(4,
        {
            { 13, ActivationFType::softplus },
        });
        const auto& lrs = tffn.get_layers();
        ASSERT_EQ(lrs.size(), 1);
        expect_layer_structure(lrs[0], 4, 13, ActivationFType::softplus);
    }

    TEST(OpennTypesTest, InitNNRandMixed)
    {
        TransparentFFN tffn(3,
        {
          { 1, ActivationFType::sigmoid },
          { 5, ActivationFType::ReLU },
          { 4, ActivationFType::softplus },
          { 2, ActivationFType::tanh },
        });
        const auto& lrs = tffn.get_layers();
        ASSERT_EQ(lrs.size(), 4);
        expect_layer_structure(lrs[0], 3, 1, ActivationFType::sigmoid);
        expect_layer_structure(lrs[1], 1, 5, ActivationFType::ReLU);
        expect_layer_structure(lrs[2], 5, 4, ActivationFType::softplus);
        expect_layer_structure(lrs[3], 4, 2, ActivationFType::tanh);
    }

    TEST(OpennTypesTest, InitNNRandDefaultActivation)
    {
        TransparentFFN tffn(3,
        {
          { 1 },
          { 5 },
          { 4 },
          { 2 },
        });
        const auto& lrs = tffn.get_layers();
        ASSERT_EQ(lrs.size(), 4);
        expect_layer_structure(lrs[0], 3, 1, ActivationFType::sigmoid);
        expect_layer_structure(lrs[1], 1, 5, ActivationFType::sigmoid);
        expect_layer_structure(lrs[2], 5, 4, ActivationFType::sigmoid);
        expect_layer_structure(lrs[3], 4, 2, ActivationFType::sigmoid);
    }

    TEST(OpennTypesTest, InitNNValuesDef)
    {
        TransparentFFN tffn_def({});
        const auto& lrs_def = tffn_def.get_layers();
        ASSERT_EQ(lrs_def.size(), 0);
    }

    TEST(OpennTypesTest, InitNNValues1)
    {
        TransparentFFN tffn({
            { ActivationFType::softplus, { {{.1, .2}, {.1, .2}, {.1, .2}}, {.1, .2, .3} } },
        });
        const auto& lrs = tffn.get_layers();
        ASSERT_EQ(lrs.size(), 1);
        expect_layer_structure(lrs[0], 2, 3, ActivationFType::softplus);
    }

    TEST(OpennTypesTest, InitNNValuesDefaultActivation)
    {
        TransparentFFN tffn({
            { .wnb={ {{.1, .2}, {.1, .2}, {.1, .2}}, {.1, .2, .3} } },
            { .wnb={ {{.1, .2, .3}},                 {.1}         } },
            { .wnb={ {{.1}, {.1}},                   {.1, .2}     } }
        });
        const auto& lrs = tffn.get_layers();
        ASSERT_EQ(lrs.size(), 3);
        expect_layer_structure(lrs[0], 2, 3, ActivationFType::sigmoid);
        expect_layer_structure(lrs[1], 3, 1, ActivationFType::sigmoid);
        expect_layer_structure(lrs[2], 1, 2, ActivationFType::sigmoid);
    }

    TEST(OpennTypesTest, InitNNValuesMixed)
    {
        TransparentFFN tffn({
            { .activation_type=ActivationFType::tanh,    .wnb={ {{.1}, {.1}, {.1}},             {.1, .2, .3} } },
            { .activation_type=ActivationFType::sigmoid, .wnb={ {{.1, .2, .3}, {.1, .2, .3}},   {.1, .2}     } },
            { .activation_type=ActivationFType::ReLU,    .wnb={ {{.1, .2}, {.1, .2}},           {.1, .2}     } }
        });
        const auto& lrs = tffn.get_layers();
        ASSERT_EQ(lrs.size(), 3);
        expect_layer_structure(lrs[0], 1, 3, ActivationFType::tanh);
        expect_layer_structure(lrs[1], 3, 2, ActivationFType::sigmoid);
        expect_layer_structure(lrs[2], 2, 2, ActivationFType::ReLU);
    }
}