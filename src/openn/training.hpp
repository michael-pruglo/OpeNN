#pragma once

#include <core/random.hpp>
#include <core/algebra.hpp>
#include <openn/openn.hpp>
#include <iostream>

using core::operator+;

namespace openn
{
    struct TrainingSample
    {
        Vec input, expected;
    };

    inline float_t cost(const Vec& output, const Vec& expected)
    {
        return core::mean_squared_eror(output, expected);
    }

    inline Vec grad(const TrainingSample& sample, FeedForwardNetwork& nn)
    {
        return {};
    }

    inline Vec batch_grad(std::vector<TrainingSample>::iterator b, std::vector<TrainingSample>::iterator e, FeedForwardNetwork& nn)
    {
        const size_t N = std::distance(b, e);
        Vec average_grad;
        for (auto sample = b; sample != e; ++sample)
        {
            const auto& gradient = grad(*sample, nn);
            average_grad = average_grad.empty() ? gradient : average_grad + gradient;
        }
        for (auto& item: average_grad)
            item /= N;
        return average_grad;
    }

    inline void process_batch(std::vector<TrainingSample>::iterator b, std::vector<TrainingSample>::iterator e, FeedForwardNetwork& nn)
    {
        //nn.apply_grad(batch_grad(b, e, nn));
    }

    inline void epoch(std::vector<TrainingSample> training_data, size_t batch_size, FeedForwardNetwork& nn)
    {
        std::shuffle(training_data.begin(), training_data.end(), core::rnd_engine);

        for (auto start = training_data.begin(), end = start; start < training_data.end(); start = end)
        {
            end = std::min(start + batch_size, training_data.end());
            process_batch(start, end, nn);
        }
    }

    /*

    epoch_gradient_descent(training_data, ...)
    {
      for (training_sample: training_data)
      {
        grad = backprop(training_sample)
        total_grad += grad
      }
      total_grad /= n
      nn.update(total_grad)
    }

    epoch_stochastic_gradient_descent(training_data, batch_size, ...)
    {
      random_shuffle(training_data)
      for (batch : training_data.split(batch_size))
        epoch_gradient_descent(batch)
    }

    //a.k.a incremental learning
    epoch_online_learning(training_data, ...)
    {
      random_shuffle(training_data)
      epoch_stochastic_gradient_descent(training_data, 1, ...) //unless we can get a significant optimization
    }

    //learning_rate is a.k.a. 'eta' - greek letter η
    //gradient is a.k.a. 'nabla' - inverted delta symbol ∇

    float_t evaluate(nn, test_data)
    {
      for ([input, output]: training_data)
        total_cost += cost(forward(input), output)
      return total_cost / n  //may be omitted, rescaling the learning rate
    }

    //make an evaluation/display wrapper that will evaluate every step and diplay graphs

    //test it on the setups used in chapter 1 http://neuralnetworksanddeeplearning.com/chap1.html

    */

}
