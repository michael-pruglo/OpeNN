#pragma once

#include <openn/opeNN.hpp>
#include <packages/nlohmann/json.hpp>

namespace openn
{
    void to_json(nlohmann::json& j, const FeedForwardNetwork& nn);
    void from_json(const nlohmann::json& j, FeedForwardNetwork& nn);

    void save_to_file(const std::string& filename, const FeedForwardNetwork& nn);
    FeedForwardNetwork load_from_file(const std::string& filename);
}