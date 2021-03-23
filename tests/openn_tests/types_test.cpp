#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>

namespace openn::types
{
    namespace activations
    {
        TEST(OpennTypesTest, ActivationSigmoid)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(activation_f(ActivationFType::SIGMOID, x), exp, 1e-11);
            };
            tst(-20.00, 0.00000000206);
            tst(-15.39, 0.00000020711);
            tst( -5.67, 0.00343601835);
            tst( -2.06, 0.11304583007);
            tst( -0.53, 0.37051688803);
            tst(  0.00, 0.5);
            tst(  0.44, 0.60825903075);
            tst(  2.11, 0.89187133324);
            tst(  6.55, 0.99857192671);
            tst( 14.61, 0.99999954819);
            tst( 20.00, 0.99999999794);
        }
    
        TEST(OpennTypesTest, ActivationSigmoidVector)
        {
            const auto tst = [](const Vec& v, const Vec& exp){
                expect_double_vec_eq(activation_f(ActivationFType::SIGMOID, v), exp, 1e-11);
            };
            tst({}, {});
            tst({2.11}, {0.89187133324});
            tst({-15.39,14.61,0.0}, {0.00000020711,0.99999954819,0.5});
        }
    
        TEST(OpennTypesTest, ActivationReLU)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(activation_f(ActivationFType::ReLU, x), exp);
            };
            tst(-20.00, 0.);
            tst( -4.00, 0.);
            tst(  0.00, 0.);
            tst(  3.14, 3.14);
            tst( 17.00, 17.);
        }
    
        TEST(OpennTypesTest, ActivationReLUVector)
        {
            const auto tst = [](const Vec& v, const Vec& exp){
                expect_double_vec_eq(activation_f(ActivationFType::ReLU, v), exp);
            };
            tst({}, {});
            tst({-2.11}, {0.});
            tst({-15.39,14.61,0.0}, {0.,14.61,0.});
        }
    
        TEST(OpennTypesTest, ActivationSoftplus)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(activation_f(ActivationFType::SOFTPLUS, x), exp, 1e-7);
            };
            tst( 1.0, 1.31326163);
            tst(-0.5, 0.474076986);
            tst( 3.4, 3.43282847042);
            tst(-2.1, 0.115519524);
            tst( 0.0, 0.693147182);
            tst(-6.5, 0.00150233845);
        }
    
        TEST(OpennTypesTest, ActivationSoftplusVector)
        {
            const auto tst = [](const Vec &v, const Vec &exp){
                expect_double_vec_eq(activation_f(ActivationFType::SOFTPLUS, v), exp, 1e-7);
            };
            tst({}, {});
            tst({3.4}, {3.43282847042});
            tst({1.0,-6.5,0.0}, {1.31326163,0.00150233845,0.693147182});
        }
    
        TEST(OpennTypesTest, ActivationTanh)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(activation_f(ActivationFType::TANH, x), exp, 1e-11);
            };
            tst(-7.80, -0.999999664235);
            tst(-0.75, -0.635148952387);
            tst( 0.00, 0.);
            tst( 1.00, 0.761594155956);
            tst( 3.14, 0.996260204946);
        }
    
        TEST(OpennTypesTest, ActivationTanhVector)
        {
            const auto tst = [](const Vec &v, const Vec &exp){
                expect_double_vec_eq(activation_f(ActivationFType::TANH, v), exp, 1e-11);
            };
            tst({}, {});
            tst({3.14}, {0.996260204946});
            tst({0.0,-7.80,1.00}, {0.,-0.999999664235,0.761594155956});
        }
    }
    
    namespace derivatives
    {
        TEST(OpennTypesTest, DerivativeSigmoid)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(derivative_f(ActivationFType::SIGMOID, x), exp, 1e-11);
            };
        }

        TEST(OpennTypesTest, DerivativeSigmoidVector)
        {
            const auto tst = [](const Vec& v, const Vec& exp){
                expect_double_vec_eq(derivative_f(ActivationFType::SIGMOID, v), exp, 1e-11);
            };
        }

        TEST(OpennTypesTest, DerivativeReLU)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(derivative_f(ActivationFType::ReLU, x), exp);
            };
        }

        TEST(OpennTypesTest, DerivativeReLUVector)
        {
            const auto tst = [](const Vec& v, const Vec& exp){
                expect_double_vec_eq(derivative_f(ActivationFType::ReLU, v), exp);
            };
        }

        TEST(OpennTypesTest, DerivativeSoftplus)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(derivative_f(ActivationFType::SOFTPLUS, x), exp);
            };

        }

        TEST(OpennTypesTest, DerivativeSoftplusVector)
        {
            const auto tst = [](const Vec &v, const Vec &exp){
                expect_double_vec_eq(derivative_f(ActivationFType::SOFTPLUS, v), exp);
            };
        }

        TEST(OpennTypesTest, DerivativeTanh)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(derivative_f(ActivationFType::TANH, x), exp);
            };

        }

        TEST(OpennTypesTest, DerivativeTanhVector)
        {
            const auto tst = [](const Vec &v, const Vec &exp){
                expect_double_vec_eq(derivative_f(ActivationFType::TANH, v), exp);
            };
        }
    }
    
    namespace nn_constructors
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