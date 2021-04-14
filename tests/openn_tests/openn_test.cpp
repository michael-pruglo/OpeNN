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

    TEST(NNComputationTest, FwdBckSmallSigmoid1)
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

    TEST(NNComputationTest, FwdBckSmallSigmoid2)
    {
        // https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html
        TransparentFFN tffn(
            {
                {},
                {{6., -2.},  {-3., 5.}},
                {{1., .25}, {-2., 2.}},
            },
            {
                {},
                {0., 0.},
                {0., 0.},
            }
        );
        constexpr float_t TOLERANCE = 1e-8;

        Gradient total_grad;
        {
            tffn.forward({ 3., 1. });
            const auto& z = tffn.get_z();
            const auto& a = tffn.get_a();
            expect_container_eq(z[1], (Vec{16., -4.}), .005);
            expect_container_eq(a[1], (Vec{.9999998874648379, .017986209962091558}), TOLERANCE);
            expect_container_eq(z[2], (Vec{1.004, -1.96}), .005);
            expect_container_eq(a[2], (Vec{0.73194171, 0.12303186}), TOLERANCE);

            const Gradient& grad = tffn.backprop({ 1., 0. }, CostFType::MEAN_SQUARED_ERROR);
            ASSERT_EQ(grad.w.shape(), (std::array<size_t, 1>{ 3 }));
            ASSERT_EQ(grad.w[1].shape(), (std::array<size_t, 2>{ 2, 2 }));
            ASSERT_EQ(grad.w[2].shape(), (std::array<size_t, 2>{ 2, 2 }));
            ASSERT_EQ(grad.b.shape(), (std::array<size_t, 1>{ 3 }));
            ASSERT_EQ(grad.b[1].shape(), (std::array<size_t, 1>{ 2 }));
            ASSERT_EQ(grad.b[2].shape(), (std::array<size_t, 1>{ 2 }));
            expect_container_eq(grad.w[2], (Matrix{ { -0.10518769048825163, -0.0018919280994576303 }, { 0.02654904571226164, 0.0004775167642113309 } }), TOLERANCE);
            expect_container_eq(grad.w[1], (Matrix{ { -5.34381483665323e-08, -1.7812716122177433e-08 }, { 0.0014201436720081408, 0.0004733812240027136 } }), TOLERANCE);

            constexpr size_t layers_count = 3;
            total_grad.w = grad.w;
            total_grad.b = Vectors({layers_count});
            for (size_t i = 0; i < layers_count; ++i)
                total_grad.b[i] = xt::zeros_like(grad.b[i]);
        }
        {
            tffn.forward({ -1., 4. });
            const auto& z = tffn.get_z();
            const auto& a = tffn.get_a();
            expect_container_eq(z[1], (Vec{-14., 23.}), .005);
            expect_container_eq(a[1], (Vec{0., 1.}), .005);
            expect_container_eq(z[2], (Vec{0.25, 2.}), .005);
            expect_container_eq(a[2], (Vec{.56, .88}), .005);

            const Gradient& grad = tffn.backprop({ 0., 1. }, CostFType::MEAN_SQUARED_ERROR);
            ASSERT_EQ(grad.w.shape(), (std::array<size_t, 1>{ 3 }));
            ASSERT_EQ(grad.w[1].shape(), (std::array<size_t, 2>{ 2, 2 }));
            ASSERT_EQ(grad.w[2].shape(), (std::array<size_t, 2>{ 2, 2 }));
            ASSERT_EQ(grad.b.shape(), (std::array<size_t, 1>{ 3 }));
            ASSERT_EQ(grad.b[1].shape(), (std::array<size_t, 1>{ 2 }));
            ASSERT_EQ(grad.b[2].shape(), (std::array<size_t, 1>{ 2 }));
            expect_container_eq(grad.w[2], (Matrix{ { 0., 0.275 }, { 0., -0.025 } }), .005);
            expect_container_eq(grad.w[1], (Matrix{ { 0., 0. }, { 0., 0. } }), .005);

            total_grad.w += grad.w;
        }

        expect_container_eq(total_grad.w[2], (Matrix{ { -0.10518769048825163, 0.2731 }, { 0.02654904571226164, -0.024 } }), 0.005);
        expect_container_eq(total_grad.w[1], (Matrix{ { -5.34381483665323e-08, 1.0712716122177433e-06 }, { 0.0014201436720081408, 0.0004733812240027136 } }), 1e-6);

        tffn.update(total_grad, 0.5);
        const auto& new_w = tffn.get_w();
        expect_container_eq(new_w[2], (Matrix{ { 1.052593845244125815, 0.11 }, { -2.01725687, 2.01595986 } }), .005);
        expect_container_eq(new_w[1], (Matrix{ { 6., -2. }, { -3.001, 4.9995 } }), .0005);

        {
            tffn.forward({ 3., 1. });
            const auto& new_res = tffn.get_a()[2];
            expect_container_eq(new_res, (Vec{0.74165987, 0.12162137}), TOLERANCE);
        }
        {
            tffn.forward({ -1., 4. });
            const auto& new_res = tffn.get_a()[2];
            expect_container_eq(new_res, (Vec{0.52811432, 0.88207988}), TOLERANCE);
        }
    }
}
