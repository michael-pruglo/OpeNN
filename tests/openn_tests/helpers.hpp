#pragma once

#include <tests/common/helpers.hpp>
#include <openn/openn.hpp>

namespace openn
{
    class TransparentFFN : public FeedForwardNetwork
    {
    public:
        using FeedForwardNetwork::FeedForwardNetwork;
        TransparentFFN(const TransparentFFN&) = default;

        [[nodiscard]] const auto& get_w() const { return w; }
        [[nodiscard]] const auto& get_b() const { return b; }
        [[nodiscard]] const auto& get_z() const { return z; }
        [[nodiscard]] const auto& get_a() const { return a; }
        [[nodiscard]] const auto& get_act_types() const { return activation_types; }
        [[nodiscard]] const auto& size() const {
            assert(w.shape()[0] == layers_count);
            assert(b.shape()[0] == layers_count);
            assert(z.shape()[0] == layers_count);
            assert(a.shape()[0] == layers_count);
            assert(activation_types.size() == layers_count);
            return layers_count;
        }
    };
}