#include <tests/common/helpers.hpp>
#include <tests/openn_tests/helpers.hpp>
#include <core/algebra.hpp>
#include <core/random.hpp>

namespace openn::types
{
    namespace nn_constructors
    {
        void check_nn_shape(
            const TransparentFFN& tffn,
            const std::vector<size_t>& layer_sizes,
            const std::vector<ActivationFType>& activations
        )
        {
            ASSERT_EQ(tffn.size(), layer_sizes.size());
            for (size_t i = 1; i < layer_sizes.size(); ++i)
            {
                const auto& prev_size = layer_sizes[i-1], curr_size = layer_sizes[i];
                const Matrix::shape_type m_shape = { curr_size, prev_size };
                const Vec   ::shape_type v_shape = { curr_size };

                EXPECT_EQ(tffn.get_w()[i].shape(), m_shape);
                EXPECT_EQ(tffn.get_b()[i].shape(), v_shape);
                EXPECT_EQ(tffn.get_z()[i].shape(), v_shape);
                EXPECT_EQ(tffn.get_a()[i].shape(), v_shape);

                EXPECT_EQ(tffn.get_act_types()[i], activations[i]);
            }
        }

        void check_nn_shape(const std::vector<size_t>& layer_sizes, const std::vector<ActivationFType>& activations)
        {
            TransparentFFN tffn(layer_sizes, activations);
            check_nn_shape(tffn, layer_sizes, activations);
        }

        void check_nn_shape(const std::vector<size_t>& layer_sizes)
        {
            TransparentFFN tffn(layer_sizes);
            check_nn_shape(
                tffn,
                layer_sizes,
                std::vector<ActivationFType>(layer_sizes.size(), ActivationFType::SIGMOID)
            );
        }


        TEST(OpennTypesTest, InitNNRandEmpty)
        {
            check_nn_shape({});
        }
        TEST(OpennTypesTest, InitNNRandUniversalAct)
        {
            check_nn_shape({1});
            check_nn_shape({7,4,1});
            check_nn_shape({3,0,5,5,6,8});
        }
        TEST(OpennTypesTest, InitNNRand)
        {
            check_nn_shape({7}, {ActivationFType::ReLU});
            check_nn_shape({3,5}, {ActivationFType::TANH, ActivationFType::SIGMOID});
            check_nn_shape({1,1,1,8,9}, {ActivationFType::SOFTPLUS, ActivationFType::ReLU, ActivationFType::SOFTPLUS, ActivationFType::TANH, ActivationFType::ReLU});
        }



        void check_val_nn_shape(
            const TransparentFFN& tffn,
            Matrixes weights,
            Vectors biases,
            std::vector<ActivationFType> activation_types
        )
        {
            std::vector<size_t> layer_sizes;
            for (const auto& b: biases)
                layer_sizes.push_back(b.size());
            check_nn_shape(tffn, layer_sizes, activation_types);
        }

        void check_val_nn_shape(Matrixes weights, Vectors biases, std::vector<ActivationFType> activation_types)
        {
            TransparentFFN tffn(weights, biases, activation_types);
            check_val_nn_shape(tffn, weights, biases, activation_types);
        }

        void check_val_nn_shape(Matrixes weights, Vectors biases)
        {
            TransparentFFN tffn(weights, biases);
            check_val_nn_shape(
                tffn,
                weights, biases,
                std::vector<ActivationFType>(biases.size(), ActivationFType::SIGMOID)
            );
        }

        TEST(OpennTypesTest, InitNNValuesDef)
        {
            check_val_nn_shape({}, {});
        }

        TEST(OpennTypesTest, InitNNValuesUniversalAct)
        {
            check_val_nn_shape(
                { core::rand_tensor<Matrix>({5,6}), core::rand_tensor<Matrix>({2,5}), core::rand_tensor<Matrix>({7,2}) },
                { core::rand_tensor<Vec>({5}), core::rand_tensor<Vec>({2}), core::rand_tensor<Vec>({7}) }
            );
        }

        TEST(OpennTypesTest, InitNNValues)
        {
            check_val_nn_shape(
                { core::rand_tensor<Matrix>({1,4}), core::rand_tensor<Matrix>({17,1}), core::rand_tensor<Matrix>({3,17}), core::rand_tensor<Matrix>({10,3}) },
                { core::rand_tensor<Vec>({1}), core::rand_tensor<Vec>({17}), core::rand_tensor<Vec>({3}), core::rand_tensor<Vec>({10}) },
                { ActivationFType::SOFTPLUS, ActivationFType::ReLU, ActivationFType::TANH, ActivationFType::SIGMOID }
            );
        }
    
    }
}