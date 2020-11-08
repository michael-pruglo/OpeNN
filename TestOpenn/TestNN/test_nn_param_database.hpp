#pragma once

#include <TestOpenn/TestNN/test_nn.hpp>
#include <unordered_map>
#include <functional>

namespace openn
{
	class ParamDatabase
	{
	public:
		enum class ParamType;
		using Filter_f = std::function<bool(ParamType)>;

	public:
		[[nodiscard]] static       std::vector<ConstructNNParam>  getEverything();
		[[nodiscard]] static       std::vector<ConstructNNParam>  getStandard();
		[[nodiscard]] static const std::vector<ConstructNNParam>& getByType(ParamType type);

		[[nodiscard]] static       std::vector<ConstructNNParam>  filter(Filter_f pred);

	private:
		static std::unordered_map<ParamType, std::vector<ConstructNNParam>> database;
	};

	enum class ParamDatabase::ParamType
	{
		NO_INSERTIONS,
		NO_INS_MULT,
		NO_INS_DIFF_ACTIVATIONS,
		INS,
		INS_CORNER_CASES,
		INS_MULT,
		INS_AT,
		INS_AT_CORNER_CASES,
		INS_AT_MULT,
		INS_DIFF_ACTIVATIONS,
		STRESS_TESTS,
	};
}
