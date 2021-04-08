#include <tests/openn_tests/helpers.hpp>

namespace openn::types
{
    TEST(OpennTest, NeuralNetComputationSmallSigmoidMultiInput)
    {
        FeedForwardNetwork nn(
            Matrixes{
                {},
                {{ -0.0169, 0.704, -0.1163 }}
            },
            Vectors{
                {},
                {0.}
            }
        );
        expect_double_vec_eq(nn.forward({22.,0.,1.}), (core::Vec{0.38}), 0.01);
        expect_double_vec_eq(nn.forward({38.,1.,1.}), (core::Vec{0.49}), 0.01);
        expect_double_vec_eq(nn.forward({26.,1.,1.}), (core::Vec{0.54}), 0.01);
        expect_double_vec_eq(nn.forward({35.,1.,1.}), (core::Vec{0.50}), 0.01);
        expect_double_vec_eq(nn.forward({35.,0.,1.}), (core::Vec{0.33}), 0.01);
        expect_double_vec_eq(nn.forward({14.,1.,1.}), (core::Vec{0.59}), 0.01);
        expect_double_vec_eq(nn.forward({25.,0.,1.}), (core::Vec{0.37}), 0.01);
        expect_double_vec_eq(nn.forward({54.,0.,1.}), (core::Vec{0.26}), 0.01);
    }

    void test_forwardpass_intermediate_results(
        TransparentFFN& tffn,
        const Vec& input,
        const std::vector<Vec>& intermediate_results,
        float_t tolerance = 0.01
    )
    {
        const size_t N = intermediate_results.size();
        ASSERT_EQ(tffn.size(), N);

        tffn.forward(input);
        for (size_t i = 0; i < N; ++i)
            expect_double_vec_eq(tffn.get_a()[i], intermediate_results[i]);
    }

    TEST(OpennTest, NeuralNetComputationSmallSigmoid)
    {
        TransparentFFN ttfn(
            {
                {},
                {{.23, .69}, {.01, .99}, {.14, .74}},
                {{.39, .97, .54}},
                {{.56}, {.89}},
            },
            {
                {},
                {.94, .49, .68},
                {.96},
                {.87, .77},
            }
        );
        test_forwardpass_intermediate_results(
            ttfn,
            {-1., 1.},
            {
                {},
                {.8, .81, .78},
                {.92},
                {.8, .83}
            }
        );
    }

    TEST(OpennTest, NeuralNetComputationSmallReLU)
    {
        TransparentFFN tffn(
            {
                {},
                {{.24, .93}, {.79, .88}},
                {{.03, .07}, {.12, .8}, {.99, .79}, {.92, .21}},
                {{.44, .28, .21, .53}, {.5, .02, .29, .68}, {.58, .22, .71, .94}},
            },
            {
                {},
                {.7, .65},
                {.06, .7, .62, .53},
                {.48, .95, .85},
            },
            {
                ActivationFType::ReLU,
                ActivationFType::ReLU,
                ActivationFType::ReLU,
            }
        );
        test_forwardpass_intermediate_results(
            tffn,
            {1.4, 1.3},
            {
                {},
                {2.25, 2.89},
                {.33, 3.28, 5.13, 3.2},
                {4.33, 4.85, 8.42}
            }
        );
    }

    TEST(OpennTest, NeuralNetComputationSmallTanh)
    {
        TransparentFFN tffn(
            {
                {},
                {{.93, -1.41}, {-1.36, -1.04}, {.23, .04}, {3.49, .6}, {-.62, 2.31}},
                {{.78, .37, -.84, .26, .4}, {2.3, 2.66, -2.45, 2.51, 2.88}},
                {{.63, -.03}, {-.84, .77}, {-.03, -.99}},
            },
            {
                {},
                {1.58, 4.14, 1.15, -.63, 1.38},
                {-1.32, -3.57},
                {.5, -.03, 1.02},
            },
            {
                ActivationFType::TANH,
                ActivationFType::TANH,
                ActivationFType::TANH,
            }
        );
        test_forwardpass_intermediate_results(
            tffn,
            {1.4, 1.3},
            {
                {},
                {.78, .7, .91, 1., 1.},
                {-0.51, 1.},
                {.148, .823, .052}
            }
        );
    }

    TEST(OpennTest, NeuralNetComputationMeduimMixed)
    {
        TransparentFFN tffn(
            {
                {},
                {{.68, .59}, {.44, .14}, {.09, .32}, {.14, .61}, {.52, .65}},
                {{.13, .96, .02, .32, .25}, {.94, .49, .5, .46, .85}, {.38, .07, -.01, .31, .01}},
                {{-.07, -.17, .45}, {-.13, .01, .4}},
                {{.6, -.11}, {.99, .33}, {.31, .69}, {.54, -.04}},
                {{.01, -.7, -.35, .45}, {.1, -.42, -.21, .27}, {.01, -.37, -.19, .24}},
            },
            {
                {},
                {.23, .43, .26, .88, .17},
                {.65, .57, .15},
                {-.12, -.18},
                {0., -.01, 0., 0.},
                {-1.7, -.82, .23}
            },
            {
                ActivationFType::TANH,
                ActivationFType::SIGMOID,
                ActivationFType::ReLU,
                ActivationFType::TANH,
                ActivationFType::SIGMOID,
            }
        );
        test_forwardpass_intermediate_results(
            tffn,
            {1.4, 1.3},
            {
                {},
                {.96, .84, .67, .95, .94},
                {.89, .97, .71},
                {0., 0.},
                {0., -.01, 0., 0.},
                {.16, .31, .56}
            }
        );
    }
}
