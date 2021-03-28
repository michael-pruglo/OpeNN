#pragma once

#include <src/projects/mnist/types.hpp>
#include <misc/resource_unpacker.hpp>
#include <openn/training.hpp>
#include <vector>
#include <string>

namespace projects::mnist
{
    using openn::NetworkTrainer;

    class TrainingEnvironment
    {
    public:
        void init_training_data(const std::string& img_file, const std::string& label_file);
        void init_test_data(const std::string& img_file, const std::string& label_file);

        void display_training_sample(size_t i) const;
        void display_validating_sample(size_t i) const;
        void display_testing_sample(size_t i) const;

    private:
        class Reader;
        class Visualizer;

    private:
        std::vector<TrainingSample> training_data, validation_data, test_data;
        std::unique_ptr<NetworkTrainer> network_trainer;
    };

    class TrainingEnvironment::Reader
    {
    public:
        std::vector<TrainingSample> read_data(const std::string& img_file, const std::string& label_file);

    private:
        void read_images(const std::string& img_file);
        void read_labels(const std::string& label_file);
        std::vector<TrainingSample> construct_training_samples() const;

    private:
        misc::IdxFileFormat image_idx, label_idx;
        size_t images_count{0}, image_pixel_count{0}, labels_count{0};
    };

    class TrainingEnvironment::Visualizer
    {
    public:
        static void display(const TrainingSample& training_sample, const std::string& id);
        static void display(const Image& image);
    private:
        static char pixel_to_char(uint8_t intensity);
    };
}


