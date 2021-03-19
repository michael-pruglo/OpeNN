#include <tests/common/helpers.hpp>
#include <openn/types.hpp>

namespace openn::types
{
    TEST(OpennTypesTest, LayerMetadataOperatorEqual)
    {
        EXPECT_EQ((LayerMetadata{}), (LayerMetadata{}));
        EXPECT_EQ((LayerMetadata{.size=1}), (LayerMetadata{.size=1}));
        EXPECT_EQ((LayerMetadata{.activation=ActivationFType::softplus}), (LayerMetadata{.activation=ActivationFType::softplus}));
        EXPECT_EQ((LayerMetadata{57,ActivationFType::tanh}), (LayerMetadata{57,ActivationFType::tanh}));
        LayerMetadata lm_empty{}, lm_s{.size=17}, lm_a{.activation=ActivationFType::ReLU}, lm_both{17,ActivationFType::tanh};
        EXPECT_EQ(lm_empty, lm_empty);
        EXPECT_EQ(lm_s, lm_s);
        EXPECT_EQ(lm_a, lm_a);
        EXPECT_EQ(lm_both, lm_both);

        EXPECT_NE(lm_empty, lm_s);
        EXPECT_NE(lm_s, lm_both);
        EXPECT_NE(lm_empty, (LayerMetadata{1}));
        EXPECT_NE(lm_empty, (LayerMetadata{.activation=ActivationFType::ReLU}));
    }

    TEST(OpennTypesTest, LayerMetadataDefaultCtor)
    {
        LayerMetadata lm_empty;
        EXPECT_EQ(lm_empty.size, 0);
        EXPECT_EQ(lm_empty.activation, ActivationFType::sigmoid);

        EXPECT_EQ(lm_empty, (LayerMetadata{}));
        EXPECT_EQ(lm_empty, (LayerMetadata{0, ActivationFType::sigmoid}));
        EXPECT_EQ((LayerMetadata{}), (LayerMetadata{0, ActivationFType::sigmoid}));
        EXPECT_EQ((LayerMetadata{0}), (LayerMetadata{0, ActivationFType::sigmoid}));
        EXPECT_EQ((LayerMetadata{.size=0}), (LayerMetadata{0, ActivationFType::sigmoid}));
        EXPECT_EQ((LayerMetadata{.activation=ActivationFType::sigmoid}), (LayerMetadata{0, ActivationFType::sigmoid}));
    }
}