#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>
#include <core/algebra.hpp>
#include <core/random.hpp>

namespace openn::types
{
    namespace activations_derivatives
    {
        using AlgebraicPred = std::function<float_t(float_t)>;

        void test_behaves_as(const AlgebraicPred& candidate, const AlgebraicPred& standard)
        {
            for (const auto& x: core::rand_vec(20, -10.0, 10.0))
                EXPECT_DOUBLE_EQ(candidate(x), standard(x));
        }

        void test_behaves_as(const std::function<Vec(const Vec&)>& candidate, const AlgebraicPred& standard)
        {
            for (const auto& vec: core::rand_matrix(20, 20, -10.0, 10.0))
            {
                const auto given = candidate(vec);
                for (size_t i = 0; i < vec.size(); ++i)
                    EXPECT_DOUBLE_EQ(given[i], standard(vec[i]));
            }
        }

        void test_act(ActivationFType type, const AlgebraicPred& standard)
        {
            test_behaves_as([type](float_t x){    return openn::activation_f(type, x); }, standard);
            test_behaves_as([type](const Vec& v){ return openn::activation_f(type, v); }, standard);
        }
        void test_der(ActivationFType type, const AlgebraicPred& standard)
        {
            test_behaves_as([type](float_t x){    return openn::derivative_f(type, x); }, standard);
            test_behaves_as([type](const Vec& v){ return openn::derivative_f(type, v); }, standard);
        }

        TEST(OpennTypesTest, ActivationSigmoid) { test_act(ActivationFType::SIGMOID, core::sigmoid); }
        TEST(OpennTypesTest, DerivativeSigmoid) { test_der(ActivationFType::SIGMOID, core::der_sigmoid); }

        TEST(OpennTypesTest, ActivationReLU) { test_act(ActivationFType::ReLU, core::relu); }
        TEST(OpennTypesTest, DerivativeReLU) { test_der(ActivationFType::ReLU, core::der_relu); }

        TEST(OpennTypesTest, ActivationSoftplus) { test_act(ActivationFType::SOFTPLUS, core::softplus); }
        TEST(OpennTypesTest, DerivativeSoftplus) { test_der(ActivationFType::SOFTPLUS, core::der_softplus); }

        TEST(OpennTypesTest, ActivationTanh) { test_act(ActivationFType::TANH, core::tanh); }
        TEST(OpennTypesTest, DerivativeTanh) { test_der(ActivationFType::TANH, core::der_tanh); }
    }

    namespace nn_constructors
    {
        using LayerT = decltype(std::declval<TransparentFFN>().get_layers().front());
        void expect_layer_structure(LayerT layer, size_t prev_size, size_t curr_size, ActivationFType activ_type)
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
                { 13, ActivationFType::SOFTPLUS },
            });
            const auto& lrs = tffn.get_layers();
            ASSERT_EQ(lrs.size(), 1);
            expect_layer_structure(lrs[0], 4, 13, ActivationFType::SOFTPLUS);
        }
    
        TEST(OpennTypesTest, InitNNRandMixed)
        {
            TransparentFFN tffn(3,
            {
                { 1, ActivationFType::SIGMOID },
                { 5, ActivationFType::ReLU },
                { 4, ActivationFType::SOFTPLUS },
                { 2, ActivationFType::TANH },
            });
            const auto& lrs = tffn.get_layers();
            ASSERT_EQ(lrs.size(), 4);
            expect_layer_structure(lrs[0], 3, 1, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[1], 1, 5, ActivationFType::ReLU);
            expect_layer_structure(lrs[2], 5, 4, ActivationFType::SOFTPLUS);
            expect_layer_structure(lrs[3], 4, 2, ActivationFType::TANH);
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
            expect_layer_structure(lrs[0], 3, 1, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[1], 1, 5, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[2], 5, 4, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[3], 4, 2, ActivationFType::SIGMOID);
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
                { ActivationFType::SOFTPLUS, {{{.1, .2}, {.1, .2}, {.1, .2}}, {.1, .2, .3} } },
            });
            const auto& lrs = tffn.get_layers();
            ASSERT_EQ(lrs.size(), 1);
            expect_layer_structure(lrs[0], 2, 3, ActivationFType::SOFTPLUS);
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
            expect_layer_structure(lrs[0], 2, 3, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[1], 3, 1, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[2], 1, 2, ActivationFType::SIGMOID);
        }
    
        TEST(OpennTypesTest, InitNNValuesMixed)
        {
            TransparentFFN tffn({
                { .activation_type=ActivationFType::TANH,    .wnb={{{.1}, {.1}, {.1}}, {.1, .2, .3} } },
                { .activation_type=ActivationFType::SIGMOID, .wnb={{{.1, .2, .3}, {.1, .2, .3}}, {.1, .2}     } },
                { .activation_type=ActivationFType::ReLU,    .wnb={ {{.1, .2}, {.1, .2}},           {.1, .2}     } }
            });
            const auto& lrs = tffn.get_layers();
            ASSERT_EQ(lrs.size(), 3);
            expect_layer_structure(lrs[0], 1, 3, ActivationFType::TANH);
            expect_layer_structure(lrs[1], 3, 2, ActivationFType::SIGMOID);
            expect_layer_structure(lrs[2], 2, 2, ActivationFType::ReLU);
        }
    }
}