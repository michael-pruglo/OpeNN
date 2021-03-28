#pragma once

#include <openn/openn.hpp>


namespace openn
{
    using core::float_t;

    // cost/loss/objective function
    enum class CostFType
    {
        MEAN_SQUARED_ERROR,
        CROSS_ENTROPY,
    };
    float_t cost_f(CostFType type, const Vec& v, const Vec& exp);
    
    struct TrainingSample
    {
        Vec input, expected;
    };
    using TrainingConstIt   = std::vector<TrainingSample>::const_iterator;
    using TrainingIt        = std::vector<TrainingSample>::iterator;

    enum class TrainingMethod
    {
        FULL_GRAD_DESCENT,
        STOCHASTIC_GRAD_DESCENT,
        ONLINE_LEARNING, //a.k.a. Incremental Learning
    };

    struct TrainingHyperParameters
    {
        size_t epochs{ 0 };
        float_t eta{ 0.0 };
        CostFType cost_f_type{ CostFType::MEAN_SQUARED_ERROR };
        TrainingMethod method{ TrainingMethod::STOCHASTIC_GRAD_DESCENT };
        size_t batch_size{ 1 };
    };

    class NetworkTrainer
    {
    public:
        NetworkTrainer() = default;

        void set_network(NeuralNetwork* network);
        void set_hyper_parameters(TrainingHyperParameters hyper_parameters);
        void set_training_data(const std::vector<TrainingSample>* training_data);
        void set_validation_data(const std::vector<TrainingSample>* validation_data);
        void set_test_data(const std::vector<TrainingSample>* test_data);
        void train(bool verbose = false);

        //evaluation
        float_t eval_average_cost();
        size_t  eval_correct_guesses();

    private:
        void epoch_full_gradient_descent       (TrainingConstIt first, TrainingConstIt last);
        void epoch_stochastic_gradient_descent (TrainingConstIt first, TrainingConstIt last, size_t batch_size);
        void epoch_online_learning             (TrainingConstIt first, TrainingConstIt last);

        void epoch_sgd_destructive(TrainingIt first, TrainingIt last, size_t batch_size);

        Gradient backprop(const TrainingSample& sample);

    private:
        NeuralNetwork* nn{ nullptr };
        TrainingHyperParameters hyper_parameters;
        const std::vector<TrainingSample> *training_data{ nullptr }, *validation_data{ nullptr }, *test_data{ nullptr };
    };


    /*

    //learning_rate is a.k.a. 'eta' - greek letter η
    //gradient is a.k.a. 'nabla' - inverted delta symbol ∇

    //test it on the setups used in chapter 1 http://neuralnetworksanddeeplearning.com/chap1.html

    */

}
