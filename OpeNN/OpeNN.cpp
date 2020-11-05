#include <cassert>
#include <iomanip>
#include "opeNN.hpp"
#include "utility.hpp"

using namespace openn;

Node::Node(size_t prev_layer_size)
	: val(openn::randd())
	, b(openn::randd())
{
	w.reserve(prev_layer_size);
	for (size_t i = 0; i < prev_layer_size; ++i)
		w.emplace_back(openn::randd());
}

NeuralNetwork::NeuralNetwork(size_t input_size, size_t output_size)
	: layers({ 
		Layer(input_size), 
		Layer(output_size, Node(input_size)) 
	})
{
}

void NeuralNetwork::addLayer(size_t layer_size)
{
	const auto& last_pre_output_idx = layers.size() - 1;
	addLayer(layer_size, last_pre_output_idx);
}

void NeuralNetwork::addLayer(size_t layer_size, size_t pos)
{
	assert(pos >= 0 && pos < layers.size());

	auto it = layers.begin() + pos;
	const auto& prev_size = pos ? (it-1)->size() : 0;
	Layer new_layer; 
	for (size_t i = 0; i < layer_size; ++i)
		new_layer.emplace_back(prev_size);

	it = layers.insert(it, new_layer);

	const auto& it_next_layer = it+1;
	for (auto& node: *it_next_layer)
		node = Node(layer_size);
}

std::ostream& openn::operator<<(std::ostream& os, const NeuralNetwork& nn)
{
	const auto& it_to_longest = std::max_element(nn.layers.begin(), nn.layers.end(), 
		[](const Layer& l1, const Layer& l2){ return l1.size()<l2.size(); }
	);
	const auto& height = it_to_longest->size();

	constexpr size_t HGAP = 5, LAYER_W = 9;
	const size_t TOTAL_W = (HGAP+LAYER_W)*nn.layers.size();

	os << std::string(TOTAL_W + HGAP, '=') << "\n";
	for (size_t i = 0; i < height; ++i)
	{
		for (const auto& layer: nn.layers)
		{
			os << std::string(HGAP, ' ');
			if (i < layer.size())
				os << "(" << std::setw(2) << layer[i].w.size() << ") " << std::fixed << std::setprecision(2) << layer[i].val;
			else 
				os << std::string(LAYER_W, ' ');
		}
		os << "\n";
	}
	os << std::string(TOTAL_W + HGAP, '=') << "\n";

	return os;
}
