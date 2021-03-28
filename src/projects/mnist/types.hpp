#pragma once

#include <src/openn/types.hpp>
#include <cstdint>

namespace projects::mnist
{
    using core::Vec;

    struct Image
    {
        Image(Vec pixels = {}, size_t height = 0, size_t width = 0)
            : pixels(std::move(pixels))
            , height(height)
            , width(width)
        {}

        size_t height, width;
        Vec pixels;
    };

    using Label = uint8_t;

    struct TrainingSample
    {
        Image image;
        Label label;
    };
}