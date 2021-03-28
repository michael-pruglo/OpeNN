#pragma once

#include <openn/openn.hpp>

namespace openn
{
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
    using TrainingDataPtr   = std::shared_ptr<const std::vector<TrainingSample>>;
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
        float_t eta{ 0.0 }; //greek letter 'η', a.k.a. learning_rate
        CostFType cost_f_type{ CostFType::MEAN_SQUARED_ERROR };
        TrainingMethod method{ TrainingMethod::STOCHASTIC_GRAD_DESCENT };
        size_t batch_size{ 1 };
    };

    class NetworkTrainer
    {
    public:
        //initialization
        virtual void    set_network            (std::shared_ptr<NeuralNetwork> network) = 0;
        virtual void    set_hyper_parameters   (TrainingHyperParameters hyper_parameters) = 0;
        virtual void    set_training_data      (TrainingDataPtr training_data) = 0;
        virtual void    set_validation_data    (TrainingDataPtr validation_data) = 0;
        virtual void    set_test_data          (TrainingDataPtr test_data) = 0;

        //training
        virtual void    train(bool verbose = false) = 0;

        //evaluation
        virtual float_t eval_average_cost() = 0;
        virtual size_t  eval_correct_guesses() = 0;

    protected:
        std::shared_ptr<NeuralNetwork> nn;
        TrainingHyperParameters hyper_parameters;
        TrainingDataPtr training_data, validation_data, test_data;
    };


    class FeedForwardNetworkTrainer : public NetworkTrainer
    {
    public:
        void    set_network            (std::shared_ptr<NeuralNetwork> network) override;
        void    set_hyper_parameters   (TrainingHyperParameters hyper_parameters) override;
        void    set_training_data      (TrainingDataPtr training_data) override;
        void    set_validation_data    (TrainingDataPtr validation_data) override;
        void    set_test_data          (TrainingDataPtr test_data) override;

        void    train(bool verbose = false) override;

        float_t eval_average_cost() override;
        size_t  eval_correct_guesses() override;

    private:
        void epoch_full_gradient_descent       (TrainingConstIt first, TrainingConstIt last);
        void epoch_stochastic_gradient_descent (TrainingConstIt first, TrainingConstIt last, size_t batch_size);
        void epoch_online_learning             (TrainingConstIt first, TrainingConstIt last);

        void epoch_sgd_destructive(TrainingIt first, TrainingIt last, size_t batch_size);

        Gradient backprop(const TrainingSample& sample);
    };


    /*

    //gradient is a.k.a. 'nabla' - inverted delta symbol ∇

    //test it on the setups used in chapter 1 http://neuralnetworksanddeeplearning.com/chap1.html

    */

}
