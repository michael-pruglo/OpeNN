#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>

namespace openn::types
{
    void test_get_layer_metadata(const std::vector<LayerMetadata>& metadata_vec)
    {
        FeedForwardNetwork nn(metadata_vec);
        const size_t N = metadata_vec.size();
        for (size_t i = 0; i < N; ++i)
        {
            EXPECT_NO_THROW(nn.get_layer_metadata(i));
            EXPECT_EQ(nn.get_layer_metadata(i), metadata_vec[i]);
        }
        EXPECT_THROW(nn.get_layer_metadata(N), std::out_of_range);
    }

    void test_get_layer_metadata_default()
    {
        FeedForwardNetwork nn_default;
        EXPECT_NO_THROW(nn_default.get_layer_metadata(0));
        EXPECT_NO_THROW(nn_default.get_layer_metadata(1));
        EXPECT_EQ(nn_default.get_layer_metadata(0), (LayerMetadata{}));
        EXPECT_EQ(nn_default.get_layer_metadata(1), (LayerMetadata{}));
        EXPECT_THROW(nn_default.get_layer_metadata(2), std::out_of_range);
    }

    TEST(OpennTest, GetLayerMetadata)
    {
        test_get_layer_metadata_default();
        test_get_layer_metadata({ LayerMetadata{} });
        test_get_layer_metadata({ {7, ActivationFType::ReLU} });
        test_get_layer_metadata({
            {16, ActivationFType::tanh},
            {42, ActivationFType::ReLU},
            {13, ActivationFType::sigmoid},
            {4, ActivationFType::softplus},
        });
    }

    TEST(OpennTest, NeuralNetComputationSmallSigmoidMultiInput)
    {
        TestableFeedForwardNetwork tnn({{3}, {1} });
        tnn.set_layer(1, {{-0.0169, 0.704, -0.1163}}, {0.});
        expect_double_vec_eq(tnn({22.,0.,1.}), (core::Vec{0.38}), 0.01);
        expect_double_vec_eq(tnn({38.,1.,1.}), (core::Vec{0.49}), 0.01);
        expect_double_vec_eq(tnn({26.,1.,1.}), (core::Vec{0.54}), 0.01);
        expect_double_vec_eq(tnn({35.,1.,1.}), (core::Vec{0.50}), 0.01);
        expect_double_vec_eq(tnn({35.,0.,1.}), (core::Vec{0.33}), 0.01);
        expect_double_vec_eq(tnn({14.,1.,1.}), (core::Vec{0.59}), 0.01);
        expect_double_vec_eq(tnn({25.,0.,1.}), (core::Vec{0.37}), 0.01);
        expect_double_vec_eq(tnn({54.,0.,1.}), (core::Vec{0.26}), 0.01);
    }

    void test_forwardpass_intermediate_results(
        const Vec& input,
        const std::vector<LayerMetadata>& nn_metadata,
        const std::vector<std::pair<Matrix, Vec>>& layers,
        const std::vector<Vec>& intermediate_results,
        float_t tolerance = 0.01
    )
    {
        for (size_t depth = 1; depth < nn_metadata.size(); ++depth)
        {
            TestableFeedForwardNetwork tnn({nn_metadata.begin(), nn_metadata.begin() + depth + 1U});
            for (size_t i = 1; i <= depth; ++i)
            {
                const auto& [weights, bias] = layers[i];
                tnn.set_layer(i, weights, bias);
            }

            expect_double_vec_eq(tnn(input), intermediate_results[depth], tolerance);
        }
    }

    TEST(OpennTest, NeuralNetComputationSmallSigmoid)
    {
        test_forwardpass_intermediate_results(
            {-1., 1.},
            {{2}, {3}, {1}, {2}},
            {
                {},
                { {{.23, .69}, {.01, .99}, {.14, .74}}, {.94, .49, .68} },
                { {{.39, .97, .54}}, {.96} },
                { {{.56}, {.89}}, {.87, .77} }
            },
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
        test_forwardpass_intermediate_results(
            {1.4, 1.3},
            {
                {2},
                {2, ActivationFType::ReLU},
                {4, ActivationFType::ReLU},
                {3, ActivationFType::ReLU}
            },
            {
                {},
                { {{.24, .93}, {.79, .88}}, {.7, .65} },
                { {{.03, .07}, {.12, .8}, {.99, .79}, {.92, .21}}, {.06, .7, .62, .53} },
                { {{.44, .28, .21, .53}, {.5, .02, .29, .68}, {.58, .22, .71, .94}}, {.48, .95, .85} },
            },
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
        test_forwardpass_intermediate_results(
            {1.4, 1.3},
            {
                {2, ActivationFType::tanh},
                {5, ActivationFType::tanh},
                {2, ActivationFType::tanh},
                {3, ActivationFType::tanh}
            },
            {
                {},
                { {{.93, -1.41}, {-1.36, -1.04}, {.23, .04}, {3.49, .6}, {-.62, 2.31}}, {1.58, 4.14, 1.15, -.63, 1.38} },
                { {{.78, .37, -.84, .26, .4}, {2.3, 2.66, -2.45, 2.51, 2.88}}, {-1.32, -3.57} },
                { {{.63, -.03}, {-.84, .77}, {-.03, -.99}}, {.5, -.03, 1.02} }
            },
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
        test_forwardpass_intermediate_results(
            {1.4, 1.3},
            {
                {2},
                {5, ActivationFType::tanh},
                {3, ActivationFType::sigmoid},
                {2, ActivationFType::ReLU},
                {4, ActivationFType::tanh},
                {3, ActivationFType::sigmoid},
            },
            {
                {},
                { {{.68, .59},{.44, .14},{.09, .32},{.14, .61},{.52, .65}}, {.23, .43, .26, .88, .17} },
                { {{.13, .96, .02, .32, .25},{.94, .49, .5, .46, .85},{.38, .07, -.01, .31, .01}}, {.65, .57, .15} },
                { {{-.07, -.17, .45},{-.13, .01, .4}}, {-.12, -.18} },
                { {{.6,-.11},{.99, .33},{.31, .69},{.54, -.04}}, {0., -.01, 0., 0.} },
                { {{.01, -.7, -.35, .45},{.1, -.42, -.21, .27},{.01, -.37, -.19, .24, .22}}, {-1.7, -.82, .23} },
            },
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
