#include <TestOpenn/TestNN/test_nn_param_database.hpp>

using namespace openn;

std::vector<ConstructNNParam> ParamDatabase::getEverything()
{
	return filter([](ParamType type) { return true; });
}

std::vector<ConstructNNParam> ParamDatabase::getStandard()
{
	return filter([](ParamType type) { return type != ParamType::STRESS_TESTS; });
}

const std::vector<ConstructNNParam>& ParamDatabase::getByType(ParamType type)
{
	return database.at(type);
}

std::vector<ConstructNNParam> ParamDatabase::filter(Filter_f pred)
{
	std::vector<ConstructNNParam> res;
	for (const auto& [param_type, param_vec] : database)
	{
		if (pred(param_type))
			std::copy(param_vec.begin(), param_vec.end(), std::back_inserter(res));
	}
	return res;
}
