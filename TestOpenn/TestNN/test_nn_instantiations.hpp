#pragma once

#include "test_nn.hpp"


namespace openn
{
	INSTANTIATE_TEST_CASE_P(
		Rand,
        NNConstructFixture,
        testing::ValuesIn(generateRandCtorParam<ConstructNNParam>(20));
	);

	INSTANTIATE_TEST_CASE_P(
		CornerCases,
        NNConstructFixture,
        testing::Values(
			ConstructNNParam{ 0, 0 },
			ConstructNNParam{ 0, 1 },
			ConstructNNParam{ 1, 0 },
			ConstructNNParam{ 1, 1 },
			ConstructNNParam{ 1, 100 },
			ConstructNNParam{ 100, 1 },
			ConstructNNParam{ 5, 1 },
			ConstructNNParam{ 1, 5 },
			ConstructNNParam{ 100, 100 }
		)
	);


	using Ins = openn::ConstructNNParam::Insertion;

	INSTANTIATE_TEST_CASE_P(
		AddUnparametrized,
        NNAddLayerFixture,
        testing::Values(
			ConstructNNParam{ 2, 2, { Ins(7) } },
			ConstructNNParam{ 2, 3, { Ins(1) } },
			ConstructNNParam{ 7, 7, { Ins(1), Ins(12) } },
			ConstructNNParam{ 7, 7, { Ins(11), Ins(1), Ins(3) } }
		)
	);
	
	INSTANTIATE_TEST_CASE_P(
		AddAt,
        NNAddLayerFixture,
        testing::Values(
			ConstructNNParam{ 7, 8, { Ins(4, 0) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1), Ins(4, 1) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1), Ins(4, 2) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1), Ins(4, 1), Ins(6, 1) } }
		)
	);

	INSTANTIATE_TEST_CASE_P(
		Rand,
        NNAddLayerFixture,
        testing::ValuesIn(generateRandCtorParam<ConstructNNParam>(100));
	);

	INSTANTIATE_TEST_CASE_P(
		AddUnparametrizedCornerCases,
        NNAddLayerFixture,
        testing::Values(
			ConstructNNParam{ 0, 0, { Ins(7) } },
			ConstructNNParam{ 0, 3, { Ins(1) } },
			ConstructNNParam{ 2, 0, { Ins(2) } },
			ConstructNNParam{ 2, 3, { Ins(0) } },
			ConstructNNParam{ 0, 0, { Ins(0) } },
			//AddLayerTestParam{ 10000, 10000, { Ins(7) } },
			//AddLayerTestParam{ 2, 2, std::vector<Ins>(20, Ins(1000)) },
			//AddLayerTestParam{ 1000, 1000, std::vector<Ins>(20, Ins(1000)) },
			ConstructNNParam{ 0, 0, { Ins(2), Ins(0), Ins(1), Ins(0) } },
			ConstructNNParam{ 0, 0, { Ins(0), Ins(0), Ins(0), Ins(0) } }
		)
	);
	
	INSTANTIATE_TEST_CASE_P(
		AddAtCornerCases,
        NNAddLayerFixture,
        testing::Values(
			ConstructNNParam{ 0, 0, { Ins(7, 0) } },
			ConstructNNParam{ 0, 3, { Ins(1, 1) } },
			ConstructNNParam{ 2, 0, { Ins(2, 0) } },
			ConstructNNParam{ 2, 3, { Ins(0, 1) } },
			ConstructNNParam{ 0, 0, { Ins(0, 1) } },
			ConstructNNParam{ 0, 0, { Ins(2, 0), Ins(0, 2), Ins(1, 1), Ins(0, 0) } },
			ConstructNNParam{ 0, 0, { Ins(0, 0), Ins(0, 2), Ins(0, 1), Ins(0, 0) } },
			//AddLayerTestParam{ 1000, 1000, std::vector<Ins>(20, Ins(1000, 0)) },
			ConstructNNParam{ 1, 1, { Ins(1, 1), Ins(1, 2), Ins(1, 3), Ins(1, 4), Ins(1, 5) } },
			ConstructNNParam{ 1, 1, { Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0) } }
		)
	);



	INSTANTIATE_TEST_CASE_P(
		Everything,
        NNJsonSerializeFixture,
        testing::Values(
			ConstructNNParam{ 0, 0, { Ins(7, 0) } },
			ConstructNNParam{ 0, 3, { Ins(1, 1) } },
			ConstructNNParam{ 2, 0, { Ins(2, 0) } },
			ConstructNNParam{ 2, 3, { Ins(0, 1) } },
			ConstructNNParam{ 0, 0, { Ins(0, 1) } },
			ConstructNNParam{ 0, 0, { Ins(2, 0), Ins(0, 2), Ins(1, 1), Ins(0, 0) } },
			ConstructNNParam{ 0, 0, { Ins(0, 0), Ins(0, 2), Ins(0, 1), Ins(0, 0) } },
			//AddLayerTestParam{ 1000, 1000, std::vector<Ins>(20, Ins(1000, 0)) },
			ConstructNNParam{ 1, 1, { Ins(1, 1), Ins(1, 2), Ins(1, 3), Ins(1, 4), Ins(1, 5) } },
			ConstructNNParam{ 1, 1, { Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0) } }
		)
	);
	
}