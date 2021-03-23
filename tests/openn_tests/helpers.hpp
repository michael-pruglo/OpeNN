#pragma once

#include <tests/common/helpers.hpp>
#include <openn/openn.hpp>

namespace openn
{
    class TransparentFFN : public FeedForwardNetwork
    {
    public:
        using FeedForwardNetwork::FeedForwardNetwork;

        const std::vector<Layer>& get_layers() const { return layers; }
    };
}