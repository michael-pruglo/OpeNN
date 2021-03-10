#pragma once

#include <unordered_map>
#include <functional>

namespace test_core
{
	template<typename Param>
	class ParamDB
	{
	public:
		using ParamType_t = int;
		using Data_t = std::unordered_map<ParamType_t, std::vector<Param>>;
		using Filter_f = std::function<bool(ParamType_t)>;

	public:
		void set_data(Data_t data) { data = std::move(data); }

		[[nodiscard]] std::vector<Param> get_everything() const;
		[[nodiscard]] std::vector<Param> get_by_type(ParamType_t type) const;
		[[nodiscard]] std::vector<Param> get_by_filter(Filter_f pred) const;

	private:
		Data_t data;
	};
}