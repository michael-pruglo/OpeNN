#include <openn/training.hpp>
#include <core/random.hpp>
#include <core/algebra.hpp>

using namespace openn;
using core::operator+;

namespace
{
    using openn::float_t;
    using CostF = std::function<float_t(const Vec&, const Vec&)>;

    const std::unordered_map<CostFType, CostF> COST_FUNCTIONS = {
        { CostFType::MEAN_SQUARED_ERROR, core::mean_squared_eror },
        { CostFType::CROSS_ENTROPY,      core::cross_entropy },
    };
}

core::float_t openn::cost_f(CostFType type, const Vec &v, const Vec &exp)
{
    const auto& f = COST_FUNCTIONS.at(type);
    return f(v, exp);
}



void FeedForwardNetworkTrainer::set_network(std::shared_ptr<NeuralNetwork> network)
{
    nn = network;
}

void FeedForwardNetworkTrainer::set_hyper_parameters(TrainingHyperParameters hyper_params)
{
    hyper_parameters = hyper_params;
}

void FeedForwardNetworkTrainer::set_training_data(TrainingDataPtr const tr_data)
{
    training_data = tr_data;
}

void FeedForwardNetworkTrainer::set_validation_data(TrainingDataPtr const val_data)
{
    validation_data = val_data;
}

void FeedForwardNetworkTrainer::set_test_data(TrainingDataPtr const tst_data)
{
    test_data = tst_data;
}

void FeedForwardNetworkTrainer::train(bool verbose)
{
    assert(nn);
    assert(training_data);

    for (size_t i = 0; i < hyper_parameters.epochs; ++i)
    {
        switch (hyper_parameters.method)
        {
        case TrainingMethod::FULL_GRAD_DESCENT:
            epoch_full_gradient_descent(training_data->begin(), training_data->end());
            break;
        case TrainingMethod::STOCHASTIC_GRAD_DESCENT:
            epoch_stochastic_gradient_descent(training_data->begin(), training_data->end(),
                hyper_parameters.batch_size);
            break;
        case TrainingMethod::ONLINE_LEARNING:
            epoch_online_learning(training_data->begin(), training_data->end());
            break;
        }
    }
}

void FeedForwardNetworkTrainer::epoch_full_gradient_descent(TrainingConstIt first, TrainingConstIt last)
{
    Gradient total_grad;
    for (auto tr_sample = first; tr_sample != last; ++tr_sample)
    {
        const auto& grad = backprop(*tr_sample);
        if (tr_sample == first)
            total_grad = grad;
        else
            total_grad += grad;
    }
    total_grad /= std::distance(first, last);

    nn->update(total_grad, hyper_parameters.eta);
}

void FeedForwardNetworkTrainer::epoch_stochastic_gradient_descent(TrainingConstIt first, TrainingConstIt last, size_t batch_size)
{
    std::vector<TrainingSample> tr_data_copy{first, last};
    epoch_sgd_destructive(tr_data_copy.begin(), tr_data_copy.end(), batch_size);
}

void FeedForwardNetworkTrainer::epoch_sgd_destructive(TrainingIt first, TrainingIt last, size_t batch_size)
{
    std::shuffle(first, last, core::rnd_engine);
    for (TrainingIt batch_start = first, batch_end; batch_start < last; batch_start = batch_end)
    {
        batch_end = std::min(batch_start + batch_size, last);
        epoch_full_gradient_descent(batch_start, batch_end);
    }
}

void FeedForwardNetworkTrainer::epoch_online_learning(TrainingConstIt first, TrainingConstIt last)
{
    epoch_stochastic_gradient_descent(first, last, 1); //unless we can get a significant speed up
}

Gradient FeedForwardNetworkTrainer::backprop(const TrainingSample& sample)
{
    throw "not implemented";
}

core::float_t FeedForwardNetworkTrainer::eval_average_cost()
{
    core::float_t total_cost = 0.0;
    for (const auto& [input, expected]: *test_data)
        total_cost += cost_f(hyper_parameters.cost_f_type, (*nn)(input), expected);
    return total_cost / test_data->size();  //may be omitted, rescaling the learning rate
}

namespace
{
    bool is_classification_problem(TrainingDataPtr test_data)
    {
        for (const auto& [_, expected]: *test_data)
        {
            const auto count = [&expected](float_t x){
                return std::count_if(expected.begin(), expected.end(), [x](auto y){ return core::is_equal(y, x); });
            };

            if (count(1.0) != 1 || count(0.0) != expected.size()-1)
                return false;
        }

        return true;
    }

    bool is_correct_guess(const Vec& given, const Vec& expected)
    {
        const auto& correct_given = std::max_element(given.begin(), given.end());
        const auto& correct_expected = std::max_element(expected.begin(), expected.end());
        return std::distance(given.begin(), correct_given) == std::distance(expected.begin(), correct_expected);
    }
}

size_t FeedForwardNetworkTrainer::eval_correct_guesses()
{
    assert(is_classification_problem(test_data));

    size_t correct_guesses = 0;
    for (const auto& [input, expected]: *test_data)
        correct_guesses += is_correct_guess((*nn)(input), expected);
    return correct_guesses;
}


