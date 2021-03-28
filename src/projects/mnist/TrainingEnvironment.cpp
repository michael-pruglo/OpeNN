#include <src/projects/mnist/TrainingEnvironment.hpp>
#include <cassert>
#include <iostream>

using namespace projects::mnist;
using misc::IdxFileFormat;
using misc::IdxFileUnpacker;

namespace
{
    constexpr size_t TRAINING_DATA_SIZE = 50'000;
    constexpr size_t VALIDATION_DATA_SIZE = 10'000;
    constexpr size_t TEST_DATA_SIZE = 10'000;
}

void TrainingEnvironment::init_training_data(const std::string& img_file, const std::string& label_file)
{
    training_data = Reader().read_data(img_file, label_file);
    assert(training_data.size() == TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);

    std::move(training_data.begin()+TRAINING_DATA_SIZE, training_data.end(), std::back_inserter(validation_data));
    training_data.resize(TRAINING_DATA_SIZE);

    assert(training_data.size() == TRAINING_DATA_SIZE);
    assert(validation_data.size() == VALIDATION_DATA_SIZE);
}

void TrainingEnvironment::init_test_data(const std::string& img_file, const std::string& label_file)
{
    test_data = Reader().read_data(img_file, label_file);
    assert(test_data.size() == TEST_DATA_SIZE);
}

void TrainingEnvironment::display_training_sample(size_t i) const
{
    Visualizer::display(training_data[i]);
}


std::vector<TrainingSample> TrainingEnvironment::Reader::read_data(const std::string& img_file, const std::string& label_file)
{
    read_images(img_file);
    read_labels(label_file);
    return construct_training_samples();
}

void TrainingEnvironment::Reader::read_images(const std::string& img_file)
{
    image_idx = IdxFileUnpacker::unpack(img_file);
    assert(image_idx.dimensions.size() == 3);

    images_count = image_idx.dimensions[0];
    image_pixel_count = image_idx.dimensions[1] * image_idx.dimensions[2];
    assert(image_idx.data.size() == images_count * image_pixel_count);
}

void TrainingEnvironment::Reader::read_labels(const std::string& label_file)
{
    label_idx = IdxFileUnpacker::unpack(label_file);
    assert(label_idx.dimensions.size() == 1);

    labels_count = label_idx.dimensions[0];
    assert(label_idx.data.size() == labels_count);
}

std::vector<TrainingSample> TrainingEnvironment::Reader::construct_training_samples() const
{
    assert(images_count == labels_count);

    const auto& N = images_count;
    std::vector<TrainingSample> data;
    data.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
        size_t beg_shift = i*image_pixel_count, end_shift = beg_shift+image_pixel_count;
        Image img(
            {image_idx.data.begin() + beg_shift, image_idx.data.begin() + end_shift},
            image_idx.dimensions[1],
            image_idx.dimensions[2]
        );
        data.push_back({img, label_idx.data[i]});
    }

    return data;
}



void TrainingEnvironment::Visualizer::display(const TrainingSample& training_sample)
{
    display(training_sample.image);
    display(training_sample.label);
}

void TrainingEnvironment::Visualizer::display(const Image& image)
{
    for (size_t i = 0; i < image.height; ++i)
    {
        for (size_t j = 0; j < image.width; ++j)
        {
            const auto& idx = i*image.width + j;
            const auto& px = image.pixels[idx];

            std::cout << px << " ";
        }
        std::cout << "\n";
    }
}

void TrainingEnvironment::Visualizer::display(const Label& label)
{
    std::cout << "Label: '" << int(label) << "'\n";
}
