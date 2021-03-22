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
        return core::norm_diff(output, expected);
    }

    inline Vec grad(const TrainingSample& sample, NeuralNetwork& nn)
    {

    }

    inline Vec batch_grad(std::vector<TrainingSample>::iterator b, std::vector<TrainingSample>::iterator e, NeuralNetwork& nn)
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

    inline void process_batch(std::vector<TrainingSample>::iterator b, std::vector<TrainingSample>::iterator e, NeuralNetwork& nn)
    {
        nn.apply_grad(batch_grad(b, e, nn));
    }

    inline void epoch(std::vector<TrainingSample> training_data, size_t batch_size, NeuralNetwork& nn)
    {
        std::shuffle(training_data.begin(), training_data.end(), core::rnd_engine);

        for (auto start = training_data.begin(), end = start; start < training_data.end(); start = end)
        {
            end = std::min(start + batch_size, training_data.end());
            process_batch(start, end, nn);
        }
    }

}
