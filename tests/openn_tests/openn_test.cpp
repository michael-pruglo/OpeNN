#include <tests/openn_tests/helpers.hpp>

namespace openn
{
    TEST(NNComputationTest, FwdSmallSigmoidMultiInput)
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
        expect_container_eq(nn.forward({22., 0., 1.}), (core::Vec{0.38}), 0.01);
        expect_container_eq(nn.forward({38., 1., 1.}), (core::Vec{0.49}), 0.01);
        expect_container_eq(nn.forward({26., 1., 1.}), (core::Vec{0.54}), 0.01);
        expect_container_eq(nn.forward({35., 1., 1.}), (core::Vec{0.50}), 0.01);
        expect_container_eq(nn.forward({35., 0., 1.}), (core::Vec{0.33}), 0.01);
        expect_container_eq(nn.forward({14., 1., 1.}), (core::Vec{0.59}), 0.01);
        expect_container_eq(nn.forward({25., 0., 1.}), (core::Vec{0.37}), 0.01);
        expect_container_eq(nn.forward({54., 0., 1.}), (core::Vec{0.26}), 0.01);
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
        for (size_t i = 1; i < N; ++i)
            expect_container_eq(tffn.get_a()[i], intermediate_results[i], tolerance);
    }

    TEST(NNComputationTest, FwdSmallSigmoid)
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

    TEST(NNComputationTest, FwdSmallReLU)
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

    TEST(NNComputationTest, FwdSmallTanh)
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

    TEST(NNComputationTest, FwdMeduimMixed)
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

    TEST(NNComputationTest, FwdBckSmallSigmoid)
    {
        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        TransparentFFN tffn(
            {
                {},
                { { .15, .2 }, { .25, .3 } },
                { { .4, .45 }, { .5, .55 } },
            },
            {
                {},
                { .35, .35 },
                { .6, .6 },
            }
        );
        constexpr float_t TOLERANCE = 1e-8;


        tffn.forward({ .05, .1 });
        const auto& z = tffn.get_z();
        const auto& a = tffn.get_a();
        EXPECT_NEAR(z[1][0], .3775, TOLERANCE);
        expect_container_eq(a[1], (Vec{.593269992, .596884378}), TOLERANCE);
        EXPECT_NEAR(z[2][0], 1.105905967, TOLERANCE);
        expect_container_eq(a[2], (Vec{.75136507, .772928465}), TOLERANCE);


        const Gradient& grad = tffn.backprop({ .01, .99 }, CostFType::MEAN_SQUARED_ERROR);
        ASSERT_EQ(grad.w.shape(), (std::array<size_t, 1>{ 3 }));
        ASSERT_EQ(grad.w[1].shape(), (std::array<size_t, 2>{ 2, 2 }));
        ASSERT_EQ(grad.w[2].shape(), (std::array<size_t, 2>{ 2, 2 }));
        ASSERT_EQ(grad.b.shape(), (std::array<size_t, 1>{ 3 }));
        ASSERT_EQ(grad.b[1].shape(), (std::array<size_t, 1>{ 2 }));
        ASSERT_EQ(grad.b[2].shape(), (std::array<size_t, 1>{ 2 }));
        expect_container_eq(grad.w[2], (Matrix{ { 2.*.082167041, 2.*.082667628 }, { 2.*-.02260254, 2.*-.022740242 } }), TOLERANCE);
        EXPECT_NEAR(grad.w[1](0,0), 2.*.000438568, TOLERANCE);
        EXPECT_NEAR(grad.b[2](0), .27699712370764428, TOLERANCE);
        EXPECT_NEAR(grad.b[1](0), .017542709220333908, TOLERANCE);

        tffn.update(grad, .25);
        const auto& w = tffn.get_w();
        expect_container_eq(w[1], (Matrix{{.149780716, .19956143},{.24975114,  .29950229}}), TOLERANCE);
        expect_container_eq(w[2], (Matrix{{.35891648,  .408666186},{.511301270, .561370121}}), TOLERANCE);
    }
}
