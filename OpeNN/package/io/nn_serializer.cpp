#include <OpeNN/package/io/nn_serializer.hpp>
#include <fstream>
#include <iomanip>

using namespace openn;

void openn::to_json(nlohmann::json& j, const Node& node)
{
	j["bias"] = node.bias;
	j["weights"] = node.w;
}

void openn::to_json(nlohmann::json& j, const NeuralNetwork& nn)
{
	for (size_t i = 0; i < nn.layers.size(); ++i)
	{
		nlohmann::json layer_json;
		layer_json["layer#"] = i;
		layer_json["nodes"] = nn.layers[i];
		j.push_back(layer_json);
	}
}

void openn::from_json(const nlohmann::json& j, Node& node)
{
	j.at("bias").get_to(node.bias);
	j.at("weights").get_to(node.w);
}

void openn::from_json(const nlohmann::json& j, Layer& layer)
{
	for (const auto& subj : j)
		layer.push_back(subj);
}

void openn::from_json(const nlohmann::json& j, NeuralNetwork& nn)
{
	nn.layers.resize(j.size());
	for (const auto& subj : j)
	{
		const size_t idx = subj.at("layer#");
		subj.at("nodes").get_to( nn.layers[idx] );
	}
}

void openn::save_to_file(const std::string& filename, const NeuralNetwork& nn)
{
	std::ofstream(filename) << std::setw(2) << nlohmann::json(nn);
}

NeuralNetwork openn::load_from_file(const std::string& filename)
{
	nlohmann::json j;
	std::ifstream(filename) >> j;
	return j;
}
