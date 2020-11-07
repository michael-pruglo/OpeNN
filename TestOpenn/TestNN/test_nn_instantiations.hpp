#pragma once

#include "test_nn.hpp"
#include "../test_nn_param_database.hpp"

namespace openn
{
	INSTANTIATE_TEST_CASE_P(
		Rand,
        NNConstructFixture,
        testing::ValuesIn(generateRandParam<ConstructNNParam>(20));
	);

	INSTANTIATE_TEST_CASE_P(
		CornerCases,
        NNConstructFixture,
        testing::ValuesIn(ParamDatabase::getByType(ParamDatabase::ParamType::NO_INSERTIONS))
	);


	INSTANTIATE_TEST_CASE_P(
		AddUnparametrized,
        NNAddLayerFixture,
        testing::ValuesIn(ParamDatabase::getByType(ParamDatabase::ParamType::INS_UNPARAMETHRIZED))
	);
	
	INSTANTIATE_TEST_CASE_P(
		AddAt,
        NNAddLayerFixture,
        testing::ValuesIn(ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT))
	);

	INSTANTIATE_TEST_CASE_P(
		Rand,
        NNAddLayerFixture,
        testing::ValuesIn(generateRandParam<ConstructNNParam>(100));
	);

	INSTANTIATE_TEST_CASE_P(
		AddUnparametrizedCornerCases,
        NNAddLayerFixture,
        testing::ValuesIn(ParamDatabase::getByType(ParamDatabase::ParamType::INS_UNPARAMETHRIZED_CORNER_CASES))
	);
	
	INSTANTIATE_TEST_CASE_P(
		AddAtCornerCases,
        NNAddLayerFixture,
        testing::ValuesIn(ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT_CORNER_CASES))
	);



	INSTANTIATE_TEST_CASE_P(
		Everything,
        NNJsonSerializeFixture,
        testing::ValuesIn(ParamDatabase::getStandard())
	);
	
}
