#pragma once

#include <src/openn/types.hpp>
#include <cstdint>

namespace projects::mnist
{
    using core::Vec;

    struct Image
    {
        Image(Vec pixels = {}) : pixels(std::move(pixels)) {}
        Vec pixels;
    };

    struct TrainingSample
    {
        Image image;
        uint8_t label;
    };
}