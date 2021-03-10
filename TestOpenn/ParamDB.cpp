#include <TestOpenn/ParamDB.hpp>

using namespace test_core;

template<typename Param>
std::vector<Param> ParamDB<Param>::get_everything() const
{
	return get_by_filter([](ParamType type) { return true; });
}

template<typename Param>
std::vector<Param> ParamDB<Param>::get_by_type(ParamDB::ParamType_t type) const
{
	return data.at(type);
}

template<typename Param>
std::vector<Param> ParamDB<Param>::get_by_filter(ParamDB::Filter_f pred) const
{
	std::vector<Param> res;
	for (const auto& [param_type, param_vec] : data)
	{
		if (pred(param_type))
			std::copy(param_vec.begin(), param_vec.end(), std::back_inserter(res));
	}
	return res;
}
