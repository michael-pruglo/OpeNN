#include "helpers.hpp"
#include "../../OpeNN/opeNN.cpp"

namespace openn
{
	TEST(NodeTest, Constructors)
	{
		for (size_t i = 0; i < 100; ++i)
			testNode( Node(i), i );
	}
	


	class NNConstructorFixture : public openn::TestWithTestcases
	{
	protected:
		std::vector<std::pair<size_t, size_t>> testcases = {
			{ 0, 0 },
			{ 0, 1 },
			{ 1, 0 },
			{ 1, 1 },
			{ 1, 2 },
			{ 1, 100 },
			{ 100, 1 },
			{ 5, 1 },
			{ 1, 5 },
			{ 100, 100 },
		};

		void addRandCase() override
		{
			testcases.emplace_back(rand_int(0,100), rand_int(0,100));
		}

		void runCase(size_t i) override
		{
			const auto& [x, y] = testcases[i];
			NeuralNetwork nn(x, y);
			testNet(nn, {x, y});
		}
	};

	TEST_F(NNConstructorFixture, Ctor00) { runCase(0); }
	TEST_F(NNConstructorFixture, Ctor0) { runCases(1, 3); }
	TEST_F(NNConstructorFixture, Ctor1x) { runCases(3, 9); }
	TEST_F(NNConstructorFixture, Ctor100100) { runCase(9); }
	TEST_F(NNConstructorFixture, CtorRnd) { runCases(10, testcases.size()); }


	
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

	class NNAddLayerFixture2 :
		public testing::TestWithParam<AddLayerTestParam>
	{};
	
	INSTANTIATE_TEST_CASE_P(
		AddUnparametrized,
        NNAddLayerFixture2,
        testing::Values(
			AddLayerTestParam{ 2, 2, {} },
			AddLayerTestParam{ 2, 3, {} }
		)
	);
	
	INSTANTIATE_TEST_CASE_P(
		AddAt,
        NNAddLayerFixture2,
        testing::Values(
			AddLayerTestParam{ 7, 8, {} },
			AddLayerTestParam{ 6, 9, {} }
		)
	);

	TEST_P(NNAddLayerFixture2, AddsLayer)
	{
		const auto& param = GetParam();
		testNet(param.createNN(), param.expectedResultSizes());
	}
}