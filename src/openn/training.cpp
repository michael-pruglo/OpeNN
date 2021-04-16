#include <openn/training.hpp>
#include <core/random.hpp>
#include <core/algebra.hpp>
#include <core/utility.hpp>

using namespace openn;



void FeedForwardNetworkTrainer::set_network(std::shared_ptr<NeuralNetwork> network)
{
    nn = network;
}

void FeedForwardNetworkTrainer::set_hyper_parameters(TrainingHyperParameters hyper_params)
{
    hyper_parameters = hyper_params;
}

void FeedForwardNetworkTrainer::set_training_data(TrainingDataPtr tr_data)
{
    training_data = tr_data;
}

void FeedForwardNetworkTrainer::set_validation_data(TrainingDataPtr val_data)
{
    validation_data = val_data;
}

void FeedForwardNetworkTrainer::set_test_data(TrainingDataPtr tst_data)
{
    test_data = tst_data;
}

void FeedForwardNetworkTrainer::train(bool verbose) const
{
    assert(nn);
    assert(training_data);

    if (verbose) FFNTrainingVisualizer::show_hyper_parameters(hyper_parameters);

    for (size_t i = 0; i < hyper_parameters.epochs; ++i)
    {
        if (!verbose)
            epoch();
        else
        {
            FFNTrainingVisualizer::show_epoch_no(i);
            epoch();
            FFNTrainingVisualizer::show_epoch_results(eval_average_cost());
        }
    }
}

void FeedForwardNetworkTrainer::epoch() const
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

void FeedForwardNetworkTrainer::epoch_full_gradient_descent(TrainingConstIt first, TrainingConstIt last) const
{
    Gradient total_grad;
    for (auto tr_sample = first; tr_sample != last; ++tr_sample)
    {
        const auto& grad = process_sample(*tr_sample);

        if (tr_sample == first)
            total_grad = grad;
        else
        {
            total_grad.w += grad.w;
            total_grad.b += grad.b;
        }
    }
    const auto& N = std::distance(first, last);
    total_grad.w /= N;
    total_grad.b /= N;

    nn->update(total_grad, hyper_parameters.eta);
}

void FeedForwardNetworkTrainer::epoch_stochastic_gradient_descent(TrainingConstIt first, TrainingConstIt last, size_t batch_size) const
{
    std::vector<TrainingSample> tr_data_copy{first, last};
    epoch_sgd_destructive(tr_data_copy.begin(), tr_data_copy.end(), batch_size);
}

void FeedForwardNetworkTrainer::epoch_sgd_destructive(TrainingIt first, TrainingIt last, size_t batch_size) const
{
    std::shuffle(first, last, core::rnd_engine);
    for (TrainingIt batch_start = first, batch_end; batch_start < last; batch_start = batch_end)
    {
        batch_end = std::min(batch_start + batch_size, last);
        epoch_full_gradient_descent(batch_start, batch_end);
    }
}

void FeedForwardNetworkTrainer::epoch_online_learning(TrainingConstIt first, TrainingConstIt last) const
{
    epoch_stochastic_gradient_descent(first, last, 1); //unless we can get a significant speed up
}

Gradient FeedForwardNetworkTrainer::process_sample(const TrainingSample& sample) const
{
    nn->forward(sample.input);
    const auto& grad = nn->backprop(sample.expected, hyper_parameters.cost_f_type);
    return grad;
}

core::float_t FeedForwardNetworkTrainer::eval_average_cost() const
{
    core::float_t total_cost = 0.0;
    for (const auto& [input, expected]: *test_data)
        total_cost += xt::sum(cost_f(hyper_parameters.cost_f_type, nn->forward(input), expected))[0];
    return total_cost / test_data->size();  //may be omitted, rescaling the learning rate
}

namespace
{
    using core::float_t;

    bool is_classification_problem(TrainingDataPtr test_data)
    {
        for (const auto& [_, expected]: *test_data)
        {
            const auto count = [&expected](float_t x){
                return std::count_if(expected.begin(), expected.end(), [x](float_t y){ return core::is_equal(y, x); });
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

size_t FeedForwardNetworkTrainer::eval_correct_guesses() const
{
    assert(is_classification_problem(test_data));

    size_t correct_guesses = 0;
    for (const auto& [input, expected]: *test_data)
        correct_guesses += is_correct_guess(nn->forward(input), expected);
    return correct_guesses;
}



void FFNTrainingVisualizer::show_hyper_parameters(const TrainingHyperParameters& hyper_parameters)
{
    std::cout << "Learning hyper parameters:\n"
              << "\t epochs: "          << hyper_parameters.epochs              << "\n"
              << "\t learning rate: "   << hyper_parameters.eta                 << "\n"
              << "\t cost function: "   << to_str(hyper_parameters.cost_f_type) << "\n"
              << "\t learning method: " << to_str(hyper_parameters.method)      << "\n";

    if (hyper_parameters.method == TrainingMethod::STOCHASTIC_GRAD_DESCENT)
        std::cout << "\t batch_size: "      << hyper_parameters.batch_size          << "\n";
}

std::string FFNTrainingVisualizer::to_str(CostFType cost_f_type)
{
    switch (cost_f_type)
    {
    case CostFType::MEAN_SQUARED_ERROR: return "mean squared error";
    case CostFType::CROSS_ENTROPY:      return "cross entropy";
    default:
        return " Error ";
    }
}

std::string FFNTrainingVisualizer::to_str(TrainingMethod method)
{
    switch (method)
    {
    case TrainingMethod::FULL_GRAD_DESCENT:       return "full gradient descent";
    case TrainingMethod::STOCHASTIC_GRAD_DESCENT: return "stochastic gradient descent";
    case TrainingMethod::ONLINE_LEARNING:         return "online learning a.k.a. incremental learning";
    default:
        return " Error ";
    }
}

void FFNTrainingVisualizer::show_epoch_no(size_t i)
{
    std::cout << "epoch " << std::setw(3) << i << "..";
}

void FFNTrainingVisualizer::show_epoch_results(float_t average_cost)
{
    std::cout << "done. Average cost: "
              << std::fixed << std::setw(10) << average_cost
              << "\n";
}
