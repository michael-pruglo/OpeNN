#pragma once

#include <tests/common/helpers.hpp>
#include <openn/types.hpp>
#include <openn/openn.hpp>

namespace openn
{
    ActivationFType rand_activation();

    class TestableFeedForwardNetwork : public FeedForwardNetwork
    {
    public:
        using FeedForwardNetwork::FeedForwardNetwork;
        void set_layer(size_t idx, Matrix w, Vec bias);
    };
}