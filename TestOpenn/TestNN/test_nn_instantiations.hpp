#pragma once

#include "test_nn.hpp"
#include "../test_nn_param_database.hpp"

namespace openn
{
	const auto RAND = generateRandParam<ConstructNNParam>;

	INSTANTIATE_TEST_CASE_P(Rand,				NNConstructFixture,	testing::ValuesIn( RAND(50) ));
	INSTANTIATE_TEST_CASE_P(CornerCases,		NNConstructFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::NO_INSERTIONS) ));

	INSTANTIATE_TEST_CASE_P(Ins,				NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS) ));
	INSTANTIATE_TEST_CASE_P(InsAt,				NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT) ));
	INSTANTIATE_TEST_CASE_P(Rand,				NNAddLayerFixture,	testing::ValuesIn( RAND(30) ));
	INSTANTIATE_TEST_CASE_P(InsCornerCases,		NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_CORNER_CASES)) );
	INSTANTIATE_TEST_CASE_P(InsAtCornerCases,	NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT_CORNER_CASES)) );

	INSTANTIATE_TEST_CASE_P(Serialize,			NNjsonFixture,		testing::ValuesIn( ParamDatabase::getStandard() ));
	INSTANTIATE_TEST_CASE_P(Rand,				NNjsonFixture,		testing::ValuesIn( RAND(10) ));
}
