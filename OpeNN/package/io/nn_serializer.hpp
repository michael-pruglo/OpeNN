#pragma once

#include "../opeNN.hpp"
#include "../../../packages/nlohmann/json.hpp"

namespace openn
{
	void to_json(nlohmann::json& j, const Node& node);
	void to_json(nlohmann::json& j, const NeuralNetwork& nn);

	void from_json(const nlohmann::json& j, Node& node);
	void from_json(const nlohmann::json& j, Layer& layer);
	void from_json(const nlohmann::json& j, NeuralNetwork& nn);
}