#pragma once

#include "test_nn.hpp"

namespace openn
{
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
}