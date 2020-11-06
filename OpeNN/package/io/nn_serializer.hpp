#pragma once

#include "../opeNN.hpp"
#include "../../../packages/nlohmann/json.hpp"

namespace openn
{
	nlohmann::json to_json(const Node& node);
	nlohmann::json to_json(const Layer& layer);
	nlohmann::json to_json(const NeuralNetwork& nn);


}