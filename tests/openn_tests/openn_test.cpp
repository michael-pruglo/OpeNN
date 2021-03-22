#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>

namespace openn::types
{
    void test_get_layer_metadata(const std::vector<LayerMetadata>& metadata_vec)
    {
        NeuralNetwork nn(metadata_vec);
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
        NeuralNetwork nn_default;
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
        TestableNeuralNetwork tnn({ {3}, {1} });
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

    void test_forward_intermediate(
        const Vec& input,
        const std::vector<LayerMetadata>& nn_metadata,
        const std::vector<std::tuple<Matrix, Vec, Vec>>& layers,
        float_t tolerance = 0.01
    )
    {
        for (size_t depth = 1; depth < nn_metadata.size(); ++depth)
        {
            TestableNeuralNetwork tnn({nn_metadata.begin(), nn_metadata.begin()+depth+1});
            for (size_t i = 1; i <= depth; ++i)
            {
                const auto& [weights, bias, _] = layers[i];
                tnn.set_layer(i, weights, bias);
            }

            const auto& exp_res = std::get<2>(layers[depth]);
            expect_double_vec_eq(tnn(input), exp_res, tolerance);
        }
    }

    TEST(OpennTest, NeuralNetComputationSmallSigmoid)
    {
        test_forward_intermediate(
            {-1.,1.},
            { {2}, {3}, {1}, {2} },
            {
                {},
                { {{.23,.69},{.01,.99},{.14,.74}}, {.94,.49,.68}, {.8,.81,.78} },
                { {{.39,.97,.54}}, {.96}, {.92} },
                { {{.56},{.89}}, {.87,.77}, {.8,.83} }
            }
        );
    }
}
