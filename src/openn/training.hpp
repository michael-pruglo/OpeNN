#pragma once

#include <OpeNN/Core/random.hpp>
#include <OpeNN/Core/algebra.hpp>
#include <OpeNN/OpeNN/opeNN.hpp>

namespace openn
{
    struct TrainingSample
    {
        Vec input, output;
    };

    inline float_t cost(const Vec& output, const Vec& expected)
    {
        return core::norm_diff(output, expected);
    }

    void process_batch(std::vector<TrainingSample>::iterator b, std::vector<TrainingSample>::iterator e, NeuralNetwork& nn)
    {

    }

    void epoch(std::vector<TrainingSample> training_data, size_t batch_size, NeuralNetwork& nn)
    {
        std::shuffle(training_data.begin(), training_data.end(), core::rnd_engine);

        const size_t batches = training_data.size()/batch_size;
        for (size_t i = 0; i < batches; ++i)
        {
            const auto start = training_data.begin() + i*batch_size;
            process_batch(start, start+batch_size, nn);
        }

        const auto last_batch_start = training_data.begin() + batches*batch_size;
        if (last_batch_start < training_data.end())
            process_batch(last_batch_start, training_data.end(), nn);
    }

}
