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

void NeuralNetwork::addLayer(size_t size)
{
	assert(layers.size() > 1);
	const auto& it_last_pre_output = layers.end() - 1;
	const Layer new_layer(size, Node(it_last_pre_output->size()));
	layers.insert(it_last_pre_output, new_layer);
}
