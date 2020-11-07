#pragma once

#include "TestNN/test_nn.hpp"
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
		INS_UNPARAMETHRIZED,
		INS_UNPARAMETHRIZED_CORNER_CASES,
		INS_AT,
		INS_AT_CORNER_CASES,
		STRESS_TESTS,
	};
}
