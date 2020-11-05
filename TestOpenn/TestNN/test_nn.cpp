#include "helpers.hpp"
#include "../../OpeNN/opeNN.cpp"

namespace openn
{
	TEST(NodeTest, Constructors)
	{
		for (size_t i = 0; i < 100; ++i)
			testNode( Node(i), i );
	}
	


	struct ConstructNNParam
	{
		size_t in_size, out_size;
	};

	class NNConstructFixture :
		public testing::TestWithParam<ConstructNNParam>
	{};

	INSTANTIATE_TEST_CASE_P(
		Construct,
        NNConstructFixture,
        testing::Values(
			ConstructNNParam{ 7, 8 },
			ConstructNNParam{ 6, 9 },
			ConstructNNParam{ 0, 0 },
			ConstructNNParam{ 0, 1 },
			ConstructNNParam{ 1, 0 },
			ConstructNNParam{ 1, 1 },
			ConstructNNParam{ 1, 2 },
			ConstructNNParam{ 1, 100 },
			ConstructNNParam{ 100, 1 },
			ConstructNNParam{ 5, 1 },
			ConstructNNParam{ 1, 5 },
			ConstructNNParam{ 100, 100 }
		)
	);

	TEST_P(NNConstructFixture, Constructs)
	{
		const auto& param = GetParam();
		testNet(
			NeuralNetwork(param.in_size, param.out_size), 
			{ param.in_size, param.out_size }
		);
	}


	
	struct AddLayerTestParam
	{
		struct Insertion {
			explicit Insertion(size_t layer_size);
			Insertion(size_t layer_size, size_t pos);

			size_t layer_size, pos; 
			bool use_pos; 
		};

		[[nodiscard]] std::vector<size_t> expectedResultSizes() const;
		[[nodiscard]] NeuralNetwork createNN() const;

		size_t init_in, init_out;
		std::vector<Insertion> insertions;
	};

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

	class NNAddLayerFixture :
		public testing::TestWithParam<AddLayerTestParam>
	{};
	
	INSTANTIATE_TEST_CASE_P(
		AddUnparametrized,
        NNAddLayerFixture,
        testing::Values(
			AddLayerTestParam{ 2, 2, {} },
			AddLayerTestParam{ 2, 3, {} }
		)
	);
	
	INSTANTIATE_TEST_CASE_P(
		AddAt,
        NNAddLayerFixture,
        testing::Values(
			AddLayerTestParam{ 7, 8, {} },
			AddLayerTestParam{ 6, 9, {} }
		)
	);

	TEST_P(NNAddLayerFixture, AddsLayer)
	{
		const auto& param = GetParam();
		testNet(param.createNN(), param.expectedResultSizes());
	}
}