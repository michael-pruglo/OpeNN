#include <src/projects/mnist/TrainingEnvironment.hpp>
#include <misc/resource_unpacker.hpp>
#include <cassert>

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
    training_data = read_data(img_file, label_file);
    assert(training_data.size() == TRAINING_DATA_SIZE+VALIDATION_DATA_SIZE);

    std::move(training_data.begin()+TRAINING_DATA_SIZE, training_data.end(), std::back_inserter(validation_data));
    training_data.resize(TRAINING_DATA_SIZE);

    assert(training_data.size() == TRAINING_DATA_SIZE);
    assert(validation_data.size() == VALIDATION_DATA_SIZE);
}

void TrainingEnvironment::init_test_data(const std::string& img_file, const std::string& label_file)
{
    test_data = read_data(img_file, label_file);
    assert(test_data.size() == TEST_DATA_SIZE);
}

std::vector<TrainingSample> TrainingEnvironment::read_data(const std::string& img_file, const std::string& label_file)
{
    IdxFileFormat image_idx = IdxFileUnpacker::unpack(img_file);
    assert(image_idx.dimensions.size() == 3);

    const auto& images_count = image_idx.dimensions[0];
    const auto& image_pixel_count = image_idx.dimensions[1] * image_idx.dimensions[2];
    assert(image_idx.data.size() == images_count * image_pixel_count);


    IdxFileFormat label_idx = IdxFileUnpacker::unpack(label_file);
    assert(label_idx.dimensions.size() == 1);

    const auto& labels_count = label_idx.dimensions[0];
    assert(label_idx.data.size() == labels_count);

    assert(images_count == labels_count);


    const auto& N = images_count;
    std::vector<TrainingSample> data;
    data.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
        size_t beg_shift = i*image_pixel_count, end_shift = beg_shift+image_pixel_count;
        Image img({image_idx.data.begin() + beg_shift, image_idx.data.begin() + end_shift});
        data.push_back({img, label_idx.data[i]});
    }

    return data;
}