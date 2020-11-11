#define _CRT_SECURE_NO_WARNINGS

#include <OpeNN/package/io/nn_serializer.hpp>
#include <OpeNN/package/io/nn_printer.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace openn;

void openn::to_json(nlohmann::json& j, const NeuralNetwork& nn)
{
	for (size_t i = 0; i < nn.layers.size(); ++i)
	{
		nlohmann::json layer_json;
		layer_json["layer#"] = i;
		layer_json["activation"] = to_string(nn.layers[i].activation);
		layer_json["w"] = nn.layers[i].w;
		layer_json["bias"] = nn.layers[i].bias;
		j.push_back(layer_json);
	}
}

void openn::from_json(const nlohmann::json& j, NeuralNetwork& nn)
{
	nn.layers.resize(j.size());
	for (const auto& subj : j)
	{
		const size_t idx = subj.at("layer#");
		nn.layers[idx].activation = string_to_activation_type(subj.at("activation"));
		subj.at("w").get_to( nn.layers[idx].w );
		subj.at("bias").get_to( nn.layers[idx].bias );
	}
}

void openn::save_to_file(const std::string& filename, const NeuralNetwork& nn)
{
	std::ofstream(filename) << std::setw(2) << nlohmann::json(nn);
}

NeuralNetwork openn::load_from_file(const std::string& filename)
{
	nlohmann::json j;
	std::ifstream in_file(filename);
	if (in_file.fail())
	{		
		const std::string msg = "\nproblem opening file \"" + filename + "\": " + strerror(errno) + "\n\n";
		std::cerr << msg;
		throw std::invalid_argument(msg);
	}

	in_file >> j;
	return j;
}
