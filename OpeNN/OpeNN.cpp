#include <cassert>
#include "opeNN.hpp"
#include "utility.hpp"

using namespace openn;

Node::Node(size_t prev_layer_size)
	: bias(openn::randd())
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

std::vector<openn::float_t> NeuralNetwork::operator()(const std::vector<float_t>& input)
{
	return forward(input, 1);
}

std::vector<openn::float_t> NeuralNetwork::forward(const std::vector<float_t>& prev, size_t idx)
{
	if (idx == layers.size())
		return prev;

	std::vector<float_t> res;
	std::transform(layers[idx].begin(), layers[idx].end(), std::back_inserter(res),
		[&prev](const Node& node) {
			return calcVal(node, prev); 
		} 
	);

	return forward(res, idx+1);
}

openn::float_t NeuralNetwork::calcVal(const Node& node, const std::vector<float_t>& prev)
{
	auto val = node.bias;
	for (size_t i = 0; i < prev.size(); ++i)
		val += prev[i] * node.w[i];
	return activationF(val);
}

openn::float_t NeuralNetwork::activationF(float_t x)
{
	return ActivationF::sigmoid(x);
}
