#include <TestOpenn/TestNN/test_nn.hpp>
#include <OpeNN/package/opeNN.cpp>

namespace openn
{
	ConstructNNParam::Insertion::Insertion(size_t layer_size_, ActivationFType activation_)
		: layer_size(layer_size_)
		, use_pos(false)
		, activation(activation_)
	{}
	ConstructNNParam::Insertion::Insertion(size_t layer_size_, size_t pos_, ActivationFType activation_)
		: layer_size(layer_size_)
		, pos(pos_)
		, use_pos(true)
		, activation(activation_)
	{}

	ConstructNNParam::Insertion ConstructNNParam::Insertion::generateRand(size_t max_allowed_pos)
	{
		const bool use_pos = randi(0, 1);
		const auto layer_size = rand_size();
		const auto where = rand_size(max_allowed_pos);
		const auto activation = static_cast<ActivationFType>(randi(0, static_cast<int>(ActivationFType::_SIZE)-1));
		return use_pos ? 
			Insertion(layer_size, where, activation) : 
			Insertion(layer_size, activation);
	}

	std::vector<size_t> ConstructNNParam::expectedResultSizes() const
	{
		std::vector<size_t> res = { init_in, init_out };
		for (const auto& ins: insertions)
		{
			const auto& position = ins.use_pos ? ins.pos : res.size() - 1;
			res.insert(res.begin()+position, ins.layer_size);
		}
		return res;
	}
	NeuralNetwork ConstructNNParam::createNN() const
	{
		NeuralNetwork nn({init_in, init_out});
		for (const auto& ins: insertions)
		{
			if (ins.use_pos)
				nn.addLayer(ins.layer_size, ins.pos, ins.activation);
			else
				nn.addLayer(ins.layer_size, ins.activation);
		}
		return nn;
	}

	ConstructNNParam ConstructNNParam::generateRand()
	{
		ConstructNNParam res{ rand_size(), rand_size(), {} };
		const size_t n = randi(0, 20);
		res.insertions.reserve(n);
		for (size_t i = 0; i < n; ++i)
			res.insertions.push_back(Insertion::generateRand(i+1));
		return res;
	}
}
