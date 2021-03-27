#pragma once

#include <src/projects/mnist/types.hpp>
#include <vector>
#include <string>

namespace projects::mnist
{
    class TrainingEnvironment
    {
    public:
        void init_training_data(const std::string& img_file, const std::string& label_file);
        void init_test_data(const std::string& img_file, const std::string& label_file);

    private:
        static std::vector<TrainingSample> read_data(const std::string& img_file, const std::string& label_file);

    private:
        std::vector<TrainingSample> training_data, validation_data, test_data;
    };
}


