#include <tests/common/helpers.hpp>
#include <core/algebra.hpp>

namespace core::algebra
{
    namespace activation_f
    {
        TEST(CoreAlgebraDeathTest, ActivationSigmoid)
        {
            expect_container_eq(
                core::sigmoid({-20.00, -15.39, -5.67, -2.06, -0.53, 0.00, 0.44, 2.11, 6.55, 14.61, 20.00}),
                {
                    0.00000000206, 0.00000020711, 0.00343601835, 0.11304583007, 0.37051688803, 0.5, 0.60825903075,
                    0.89187133324, 0.99857192671, 0.99999954819, 0.99999999794
                },
                1e-11
            );
        }

        TEST(CoreAlgebraDeathTest, ActivationReLU)
        {
            expect_container_eq(
                core::relu({-20.00, -4.00, 0.00, 3.14, 17.00}),
                {0., 0., 0., 3.14, 17.}
            );
        }

        TEST(CoreAlgebraDeathTest, ActivationSoftplus)
        {
            expect_container_eq(
                core::softplus({1.0, -0.5, 3.4, -2.1, 0.0, -6.5}),
                {1.31326163, 0.474076986, 3.43282847042, 0.115519524, 0.693147182, 0.00150233845},
                1e-7
            );
        }

        TEST(CoreAlgebraDeathTest, ActivationTanh)
        {
            expect_container_eq(
                core::tanh({-7.80, -0.75, 0.00, 1.00, 3.14}),
                {-0.999999664235, -0.635148952387, 0., 0.761594155956, 0.996260204946},
                1e-11
            );
        }
    }

    namespace derivative_f
    {
        TEST(CoreAlgebraDeathTest, DerivativeSigmoid)
        {
            expect_container_eq(
                core::der_sigmoid({-5.0, -1.4, -0.3, 0.0, .6, 1.7, 19.0}),
                {.0066480567, .1586849, .24445831, 0.25, .22878424, .13060575, 5.602796e-9},
                1e-8
            );
        }

        TEST(CoreAlgebraDeathTest, DerivativeReLU)
        {

            expect_container_eq(
                core::der_relu({-17.45, -2.5, -1e-11, 0., 1e-11, 10., 182.}),
                {0., 0., 0., 1., 1., 1., 1.}
            );
        }

        TEST(CoreAlgebraDeathTest, DerivativeSoftplus)
        {
            expect_container_eq(
                core::der_softplus({-20.00, -15.39, -5.67, -2.06, -0.53, 0.00, 0.44, 2.11, 6.55, 14.61, 20.00}),
                {
                    0.00000000206, 0.00000020711, 0.00343601835, 0.11304583007, 0.37051688803, 0.5, 0.60825903075,
                    0.89187133324, 0.99857192671, 0.99999954819, 0.99999999794
                },
                1e-11
            );
        }

        TEST(CoreAlgebraDeathTest, DerivativeTanh)
        {
            expect_container_eq(
                core::der_tanh({-5.0, -1.4, -0.3, 0.0, .6, 1.7, 19.0}),
                {1.815832e-4, 0.21615246, 0.91513696, 1.0, .71157776, .12500987, 2.220446e-16},
                1e-8
            );
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
