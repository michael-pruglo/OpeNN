#include "test_nn.hpp"

namespace openn
{
	AddLayerTestParam::Insertion::Insertion(size_t layer_size_)
		: layer_size(layer_size_)
		, use_pos(false)
	{}
	AddLayerTestParam::Insertion::Insertion(size_t layer_size_, size_t pos_)
		: layer_size(layer_size_)
		, pos(pos_)
		, use_pos(true)
	{}
	
	std::vector<size_t> AddLayerTestParam::expectedResultSizes() const
	{
		std::vector<size_t> res = { init_in, init_out };
		for (const auto& ins: insertions)
		{
			const size_t position = ins.use_pos ? ins.pos : res.size() - 1;
			res.insert(res.begin()+position, ins.layer_size);
		}
		return res;
	}
	NeuralNetwork AddLayerTestParam::createNN() const
	{
		NeuralNetwork nn(init_in, init_out);
		for (const auto& ins: insertions)
		{
			if (ins.use_pos)
				nn.addLayer(ins.layer_size, ins.pos);
			else
				nn.addLayer(ins.layer_size);
		}
		return nn;
	}
}