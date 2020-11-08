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

	constexpr openn::float_t W_MIN = -10.0, W_MAX = 10.0;
}

Node::Node(size_t prev_layer_size)
	: bias(openn::randd(W_MIN, W_MAX))
{
	generative_construct(w, prev_layer_size, []{ return openn::randd(W_MIN, W_MAX); });
}

void Node::resetWeight(size_t prev_layer_size)
{
	generative_resize(w, prev_layer_size, []{ return openn::randd(W_MIN, W_MAX); });
}

std::unordered_map<ActivationFType, ActivationF> Layer::activation_functions = {
	{ ActivationFType::ReLU,		[](float_t x) -> float_t { return std::max(0., x); } },
	{ ActivationFType::sigmoid,		[](float_t x) -> float_t { return 1. / (1. + std::exp(-x)); } },
	{ ActivationFType::softplus,	[](float_t x) -> float_t { return std::log(1. + std::exp(x)); } },
	{ ActivationFType::tanh,		[](float_t x) -> float_t { return std::tanh(x); } },
};

Layer::Layer(size_t layer_size, size_t prev_layer_size, ActivationFType activation_)
	: activation(activation_)
{
	generative_construct(*this, layer_size,
		[prev_layer_size]{ return Node(prev_layer_size); }
	);
}

openn::float_t Layer::activation_f(float_t x) const
{
	return activation_functions.at(activation)(x);
}

void Layer::resetWeights(size_t prev_layer_size)
{
	for (auto& node : *this)
		node.resetWeight(prev_layer_size);
}

LayerStructure Layer::getLayerStructure() const
{
	return { size(), activation };
}

LayerStructure::LayerStructure(size_t size_, ActivationFType activation_)
	: size(size_)
	, activation(activation_)
{
}

NeuralNetwork::NeuralNetwork(const std::vector<LayerStructure>& nn_structure)
{
	for (size_t i = 0; i < nn_structure.size(); ++i)
	{
		const auto& prev_size = i ? nn_structure[i-1].size : 0;
		layers.emplace_back(nn_structure[i].size, prev_size, nn_structure[i].activation);
	}
}

void NeuralNetwork::addLayer(size_t layer_size, ActivationFType activation)
{
	const auto& last_pre_output_idx = layers.size() - 1;
	addLayer(layer_size, last_pre_output_idx, activation);
}

void NeuralNetwork::addLayer(size_t layer_size, size_t pos, ActivationFType activation)
{
	assert(pos >= 0 && pos < layers.size());

	auto it = layers.begin() + pos;
	const auto& prev_size = pos ? (it-1)->size() : 0;
	it = layers.insert(it, Layer(layer_size, prev_size, activation));

	(it+1)->resetWeights(layer_size);
}

std::vector<LayerStructure> NeuralNetwork::getNNStructure() const
{
	std::vector<LayerStructure> res;
	for (const auto& layer: layers)
		res.emplace_back(layer.getLayerStructure());
	return res;
}

std::vector<openn::float_t> NeuralNetwork::operator()(const std::vector<float_t>& input) const
{
	return _forward(input, 1);
}

std::vector<openn::float_t> NeuralNetwork::_forward(const std::vector<float_t>& prev, size_t idx) const
{
	if (idx == layers.size())
		return prev;

	std::vector<float_t> res;
	std::transform(layers[idx].begin(), layers[idx].end(), std::back_inserter(res),
		[&prev, idx, this](const Node& node) {
			return layers[idx].activation_f(_calcVal(node, prev)); 
		} 
	);

	return _forward(res, idx+1);
}

openn::float_t NeuralNetwork::_calcVal(const Node& node, const std::vector<float_t>& prev)
{
	auto val = node.bias;
	for (size_t i = 0; i < prev.size(); ++i)
		val += prev[i] * node.w[i];
	return val;
}





bool openn::operator==(const Node& n1, const Node& n2)
{
	return float_eq(n1.bias, n2.bias) && n1.w == n2.w;
}

bool openn::operator==(const NeuralNetwork& nn1, const NeuralNetwork& nn2)
{
	return nn1.layers == nn2.layers;
}
