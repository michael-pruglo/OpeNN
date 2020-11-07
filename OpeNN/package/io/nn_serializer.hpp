#pragma once

#include <OpeNN/package/opeNN.hpp>
#include <packages/nlohmann/json.hpp>

namespace openn
{
	void to_json(nlohmann::json& j, const Node& node);
	void to_json(nlohmann::json& j, const NeuralNetwork& nn);

	void from_json(const nlohmann::json& j, Node& node);
	void from_json(const nlohmann::json& j, Layer& layer);
	void from_json(const nlohmann::json& j, NeuralNetwork& nn);

	void save_to_file(const std::string& filename, const NeuralNetwork& nn);
	NeuralNetwork load_from_file(const std::string& filename);
}