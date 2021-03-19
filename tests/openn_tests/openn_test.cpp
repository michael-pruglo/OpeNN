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

    TEST(OpennTest, NeuralNetComputation)
    {
        {
            TestableNeuralNetwork tnn({ {2}, {3} });
            tnn.set_layer(1, {{1.,2.},{3.,4.},{5.,6.}}, {1.,2.,3.});
            const core::Vec input{1.,2.}, expected_output{ 0.99752737684336534,0.99999773967570205,0.99999999793884631 };
            expect_double_vec_eq(tnn(input), expected_output);
        }
    }
}
