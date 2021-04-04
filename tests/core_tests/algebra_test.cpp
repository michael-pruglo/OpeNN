#include <tests/common/helpers.hpp>
#include <core/algebra.hpp>

namespace core::algebra
{
    using core::float_t;

    float_t rand_float_t()
    {
        return rand()%9001/1000.0;
    }

    namespace activation_f
    {
        TEST(CoreAlgebraDeathTest, ActivationSigmoid)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::sigmoid(x), exp, 1e-11);
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

        TEST(CoreAlgebraDeathTest, ActivationReLU)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(core::relu(x), exp);
            };
            tst(-20.00, 0.);
            tst( -4.00, 0.);
            tst(  0.00, 0.);
            tst(  3.14, 3.14);
            tst( 17.00, 17.);
        }

        TEST(CoreAlgebraDeathTest, ActivationSoftplus)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::softplus(x), exp, 1e-7);
            };
            tst( 1.0, 1.31326163);
            tst(-0.5, 0.474076986);
            tst( 3.4, 3.43282847042);
            tst(-2.1, 0.115519524);
            tst( 0.0, 0.693147182);
            tst(-6.5, 0.00150233845);
        }

        TEST(CoreAlgebraDeathTest, ActivationTanh)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::tanh(x), exp, 1e-11);
            };
            tst(-7.80, -0.999999664235);
            tst(-0.75, -0.635148952387);
            tst( 0.00, 0.);
            tst( 1.00, 0.761594155956);
            tst( 3.14, 0.996260204946);
        }
    }

    namespace derivative_f
    {
        TEST(CoreAlgebraDeathTest, DerivativeSigmoid)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::der_sigmoid(x), exp, 1e-8);
            };
            tst(-5.0, .0066480567);
            tst(-1.4, .1586849);
            tst(-0.3, .24445831);
            tst(0.0, 0.25);
            tst(.6, .22878424);
            tst(1.7, .13060575);
            tst(19.0, 5.602796e-9);
        }

        TEST(CoreAlgebraDeathTest, DerivativeReLU)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(core::der_relu(x), exp);
            };
            tst(-17.45, 0.);
            tst(-2.5, 0.);
            tst(-1e-11, 0.);
            tst(0., 1.);
            tst(1e-11, 1.);
            tst(10., 1.);
            tst(182., 1.);
        }

        TEST(CoreAlgebraDeathTest, DerivativeSoftplus)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::der_softplus(x), exp, 1e-11);
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

        TEST(CoreAlgebraDeathTest, DerivativeTanh)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::der_tanh(x), exp, 1e-8);
            };
            tst(-5.0, 1.815832e-4);
            tst(-1.4, 0.21615246);
            tst(-0.3, 0.91513696);
            tst(0.0, 1.0);
            tst(.6, .71157776);
            tst(1.7, .12500987);
            tst(19.0, 2.220446e-16);
        }
    }

    namespace cost_f
    {
        TEST(CoreAlgebraDeathTest, MSESmall)
        {
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{}, core::Vec{}), 0.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{7.}, core::Vec{7.}), 0.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{2., 5., 3.}, core::Vec{1., 7., 0.}), 14.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{-2., -5., -3.}, core::Vec{-1., -7., 0.}), 14.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{-5., 0., -1., 7.}, core::Vec{-4., 3., -5., 6.}), 27.);
        }

        TEST(CoreAlgebraDeathTest, MSEDeath)
        {
            // the vectors must have equal length
            EXPECT_DEATH(core::mean_squared_eror(core::Vec{}, core::Vec{1.}), "");
            EXPECT_DEATH(core::mean_squared_eror(core::Vec{1., 2., 3.}, core::Vec{1.}), "");
            EXPECT_DEATH(core::mean_squared_eror(core::Vec{1., 2.}, core::Vec{1., 2., 3.}), "");
        }

        TEST(CoreAlgebraDeathTest, CrossEntropySmall)
        {
            EXPECT_DOUBLE_EQ(core::cross_entropy(core::Vec{}, core::Vec{}), 0.);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.1}), 0.32508297, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.2}), 0.54480543, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.3}), 0.76452789, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.4}), 0.98425035, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.9}), 2.08286264, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{1.9}), 4.28008721, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{-0.1}), -0.11436194, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1,0.999}, core::Vec{0.,1.}), 0.1063610203, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{.001,.5,.7,.9,.75}, core::Vec{0.,.5,.3,.2,44.}), -43.44557503, 1e-8);
        }

        TEST(CoreAlgebraDeathTest, CrossEntropyDeath)
        {
            // the vectors must have equal length
            EXPECT_DEATH(core::cross_entropy(core::Vec{}, core::Vec{1.}), "");
            EXPECT_DEATH(core::cross_entropy(core::Vec{1., 2., 3.}, core::Vec{1.}), "");
            EXPECT_DEATH(core::cross_entropy(core::Vec{1., 2.}, core::Vec{1., 2., 3.}), "");
        }
    }
}
