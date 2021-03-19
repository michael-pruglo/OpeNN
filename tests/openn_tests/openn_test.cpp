#include <tests/common/helpers.hpp>
#include <openn/openn.hpp>

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

    TEST(OpennTest, GetLayerMetadata)
    {
        NeuralNetwork nn_default;
        EXPECT_NO_THROW(nn_default.get_layer_metadata(0));
        EXPECT_NO_THROW(nn_default.get_layer_metadata(1));
        EXPECT_EQ(nn_default.get_layer_metadata(0), (LayerMetadata{}));
        EXPECT_EQ(nn_default.get_layer_metadata(1), (LayerMetadata{}));
        EXPECT_THROW(nn_default.get_layer_metadata(2), std::out_of_range);

        test_get_layer_metadata({ LayerMetadata{} });
        test_get_layer_metadata({ {7, ActivationFType::ReLU} });
        test_get_layer_metadata({
            {16, ActivationFType::tanh},
            {42, ActivationFType::ReLU},
            {13, ActivationFType::sigmoid},
            {4, ActivationFType::softplus},
        });
    }
}





//#include <GTest/TestOpenn/test_nn.hpp>
//#include <OpeNN/OpeNN/opeNN.cpp>
//
//namespace openn
//{
//
//    ConstructNNParam::InsLayer::InsLayer(size_t layer_size_, ActivationFType activation_)
//        : layer_size(layer_size_)
//        , use_pos(false)
//        , activation(activation_)
//    {}
//    ConstructNNParam::InsLayer::InsLayer(size_t layer_size_, size_t pos_, ActivationFType activation_)
//        : layer_size(layer_size_)
//        , pos(pos_)
//        , use_pos(true)
//        , activation(activation_)
//    {}
//
//    ConstructNNParam::InsLayer ConstructNNParam::InsLayer::generateRand(size_t max_allowed_pos)
//    {
//        const bool use_pos = rand_i(0, 1);
//        const auto layer_size = rand_size();
//        const auto where = rand_size(max_allowed_pos);
//        const auto activation = rand_activation();
//        return use_pos ?
//            InsLayer(layer_size, where, activation) :
//            InsLayer(layer_size, activation);
//    }
//
//
//    ConstructNNParam::ConstructNNParam(
//        std::vector<LayerMetadata> nn_structure_,
//        std::vector<InsLayer> additional_insertions_
//    )
//        : nn_structure(std::move(nn_structure_))
//        , additional_insertions(std::move(additional_insertions_))
//    {
//    }
//
//    std::vector<LayerMetadata> ConstructNNParam::expectedResultStructure() const
//    {
//        std::vector<LayerMetadata> res = nn_structure;
//        for (const auto& ins: additional_insertions)
//        {
//            const auto& position = ins.use_pos ? ins.pos : res.size() - 1;
//            res.insert(res.begin()+position, LayerMetadata(ins.layer_size, ins.activation));
//        }
//        return res;
//    }
//    NeuralNetwork ConstructNNParam::createNN() const
//    {
//        NeuralNetwork nn(nn_structure);
//        for (const auto& ins: additional_insertions)
//        {
//            if (ins.use_pos)
//                nn.addLayer(ins.layer_size, ins.pos, ins.activation);
//            else
//                nn.addLayer(ins.layer_size, ins.activation);
//        }
//        return nn;
//    }
//
//    ConstructNNParam ConstructNNParam::generateRand()
//    {
//        std::vector<LayerMetadata> res_struct;
//        generative_construct(res_struct, rand_size()+2,
//            []{ return LayerMetadata{ rand_size(), rand_activation() }; }
//        );
//        std::vector<InsLayer> add_ins;
//        generative_construct(add_ins, rand_size(),
//            [&add_ins, &res_struct] {
//                const auto& curr_size = res_struct.size() + add_ins.size();
//                return InsLayer::generateRand(curr_size-1);
//            }
//        );
//        return { res_struct, add_ins };
//    }
//
//}
