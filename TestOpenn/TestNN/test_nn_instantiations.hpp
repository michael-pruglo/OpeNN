#pragma once

#include <TestOpenn/TestNN/test_nn.hpp>
#include <TestOpenn/TestNN/test_nn_param_database.hpp>

namespace openn
{
	const auto RAND = generateRandParam<ConstructNNParam>;

	INSTANTIATE_TEST_CASE_P(Rand,				NNConstructFixture,	testing::ValuesIn( RAND(50) ));
	INSTANTIATE_TEST_CASE_P(CornerCases,		NNConstructFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::NO_INSERTIONS) ));
	INSTANTIATE_TEST_CASE_P(Mult,				NNConstructFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::NO_INS_MULT) ));
	INSTANTIATE_TEST_CASE_P(DiffActivations,	NNConstructFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::NO_INS_DIFF_ACTIVATIONS) ));

	INSTANTIATE_TEST_CASE_P(Ins,				NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS) ));
	INSTANTIATE_TEST_CASE_P(InsMult,			NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_MULT) ));
	INSTANTIATE_TEST_CASE_P(InsAt,				NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT) ));
	INSTANTIATE_TEST_CASE_P(InsAtMult,			NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT_MULT) ));
	INSTANTIATE_TEST_CASE_P(Rand,				NNAddLayerFixture,	testing::ValuesIn( RAND(30) ));
	INSTANTIATE_TEST_CASE_P(InsCornerCases,		NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_CORNER_CASES)) );
	INSTANTIATE_TEST_CASE_P(InsAtCornerCases,	NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_AT_CORNER_CASES)) );
	INSTANTIATE_TEST_CASE_P(DiffActivations,	NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::INS_DIFF_ACTIVATIONS)) );
	INSTANTIATE_TEST_CASE_P(DISABLED_Stress,	NNAddLayerFixture,	testing::ValuesIn( ParamDatabase::getByType(ParamDatabase::ParamType::STRESS_TESTS)) );

	INSTANTIATE_TEST_CASE_P(DISABLED_Serialize,	NNjsonFixture,		testing::ValuesIn( ParamDatabase::getStandard() ));
	INSTANTIATE_TEST_CASE_P(DISABLED_Rand,		NNjsonFixture,		testing::ValuesIn( RAND(10) ));
}
	INSTANTIATE_TEST_CASE_P(Forward,			NNForwardFixture,	testing::Values( 
		InpOutpNNParam{ { 0.888, -0.49 }, { 0.74 }, 0.01, "../resources/tiny_nn_01.json" },
		InpOutpNNParam{ { 0.0, 0.0 }, { 0.5835 }, 0.0025, "../resources/tiny_nn_02.json" },
		InpOutpNNParam{ { 1.4, 1.3 }, { 0.8, 0.83, 0.88 }, 0.005, "../resources/tiny_nn_03.json" },
		InpOutpNNParam{ { 1.4, 1.3 }, { 0.85, 0.9, 0.93 }, 0.005, "../resources/tiny_nn_04.json" },
		InpOutpNNParam{ { 1.4, 1.3 }, { 1.0, 0.97, 1.0 }, 0.005, "../resources/tiny_nn_05.json" }
	));
