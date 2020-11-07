#include <OpeNN/package/opeNN.hpp>
#include <OpeNN/package/utility.hpp>
#include <cassert>

using namespace openn;

namespace
{
	template<typename T, typename Generator>
	void generative_append(std::vector<T>& v, size_t extra_amount, const Generator& gen)
	{
		v.reserve(v.size() + extra_amount);
		for (size_t i = 0; i < extra_amount; ++i)
			v.emplace_back(gen());
	}

	template<typename T, typename Generator>
	void generative_construct(std::vector<T>& v, size_t size, const Generator& gen)
	{
		generative_append(v, size, gen);
	}

	template<typename T, typename Generator>
	void generative_resize(std::vector<T>& v, size_t amount, const Generator& gen)
	{
		if (v.size() > amount)
			v.resize(amount);
		else
			generative_append(v, amount - v.size(), gen);
	}

}

Node::Node(size_t prev_layer_size)
	: bias(openn::randd())
{
	generative_construct(w, prev_layer_size, []{ return openn::randd(); });
}

void Node::resetWeight(size_t prev_layer_size)
{
	generative_resize(w, prev_layer_size, []{ return openn::randd(); });
}


Layer::Layer(size_t layer_size, size_t prev_layer_size)
{
	generative_construct(*this, layer_size,
		[prev_layer_size]{ return Node(prev_layer_size); }
	);
}

void Layer:: resetWeights(size_t prev_layer_size)
{
	for (auto& node : *this)
		node.resetWeight(prev_layer_size);
}

NeuralNetwork::NeuralNetwork(size_t input_size, size_t output_size)
	: layers({ 
		Layer(input_size), 
		Layer(output_size, input_size) 
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
	it = layers.insert(it, Layer(layer_size, prev_size));

	(it+1)->resetWeights(layer_size);
}

std::vector<openn::float_t> NeuralNetwork::operator()(const std::vector<float_t>& input) const
{
	return _forward(input, 1);
}

#include <iostream>
#include <iomanip>
std::vector<openn::float_t> NeuralNetwork::_forward(const std::vector<float_t>& prev, size_t idx) const
{
	if (idx == layers.size())
		return prev;

	std::vector<float_t> res;
	std::transform(layers[idx].begin(), layers[idx].end(), std::back_inserter(res),
		[&prev, idx](const Node& node) {
			std::cerr << "\t(" << idx << ")";
			return _calcVal(node, prev); 
		} 
	);

	std::cerr << "_forward(" << idx << ") : " << res << "\n";

	return _forward(res, idx+1);
}

openn::float_t NeuralNetwork::_calcVal(const Node& node, const std::vector<float_t>& prev)
{
	auto val = node.bias;
	std::cerr << std::setprecision(2) << node.bias;
	for (size_t i = 0; i < prev.size(); ++i)
	{
		std::cerr << " + " << prev[i] << "*" << node.w[i];
		val += prev[i] * node.w[i];
	}
	std::cerr << "  = " << val << " => " << activationF(val) << "\n";
	return activationF(val);
}

openn::float_t NeuralNetwork::activationF(float_t x)
{
	return ActivationF::sigmoid(x);
}





bool openn::operator==(const Node& n1, const Node& n2)
{
	return n1.bias == n2.bias && n1.w == n2.w;
}

bool openn::operator==(const NeuralNetwork& nn1, const NeuralNetwork& nn2)
{
	return nn1.layers == nn2.layers;
}
