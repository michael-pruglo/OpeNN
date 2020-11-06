#include "nn_serializer.hpp"

using namespace openn;

nlohmann::json openn::to_json(const Node& node)
{
	nlohmann::json res;
	res["bias"] = node.bias;
	res["weights"] = node.w;
	return res;
}

nlohmann::json openn::to_json(const Layer& layer)
{
	nlohmann::json res;
	for (const auto& node: layer)
		res.push_back(to_json(node));
	return res;
}

nlohmann::json openn::to_json(const NeuralNetwork& nn)
{
	nlohmann::json res;
	for (size_t i = 0; i < nn.layers.size(); ++i)
	{
		nlohmann::json layer_json;
		layer_json["layer#"] = i;
		layer_json["nodes"] = to_json(nn.layers[i]);
		res.push_back(layer_json);
	}
	return res;
}