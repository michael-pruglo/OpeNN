#pragma once

#include <src/openn/types.hpp>
#include <cstdint>

namespace projects::mnist
{
    struct Image
    {
        Image(std::vector<uint8_t> pixels = {}, size_t height = 0, size_t width = 0)
            : pixels(std::move(pixels))
            , height(height)
            , width(width)
        {}

        size_t height, width;
        std::vector<uint8_t> pixels;
    };

    using Label = uint8_t;

    struct TrainingSample
    {
        Image image;
        Label label;
    };
}