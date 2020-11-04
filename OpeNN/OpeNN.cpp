#include <cassert>
#include <algorithm>
#include "opeNN.hpp"
#include "utility.hpp"

using namespace openn;

Node::Node(size_t prev_layer_size)
	: val(openn::randd())
	, w(prev_layer_size)
	, b(openn::randd())
{
	std::generate_n(w.begin(), prev_layer_size, openn::randd);
}

NeuralNetwork::NeuralNetwork(size_t input_size, size_t output_size)
	: layers({ 
		Layer(input_size, Node(0)), 
		Layer(output_size, Node(input_size)) 
	})
{
}

void NeuralNetwork::addLayer(size_t layer_size)
{
	const auto& last_pre_output_idx = layers.size() - 2;
	addLayer(layer_size, last_pre_output_idx);
}

void NeuralNetwork::addLayer(size_t layer_size, size_t pos)
{
	assert(pos >= 0 && pos < layers.size()-1);

	auto it = layers.begin() + pos;
	const Layer new_layer( layer_size, Node(it->size()) );
	it = layers.insert(it, new_layer);

	const auto& it_next_layer = it+1;
	for (auto& node: *it_next_layer)
		node = Node(layer_size);
}
