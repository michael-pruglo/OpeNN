#include <iterator>

using namespace test_core;

template<typename T>
std::vector<T> Database<T>::get_everything() const
{
    return get_by_filter([](ParamType_t type) { return true; });
}

template<typename T>
std::vector<T> Database<T>::get_by_type(Database::ParamType_t type) const
{
    return data.at(type);
}

template<typename T>
std::vector<T> Database<T>::get_by_filter(Database::Filter_f pred) const
{
    std::vector<T> res;
    for (const auto& [param_type, param_vec] : data)
    {
        if (pred(param_type))
            std::copy(param_vec.begin(), param_vec.end(), std::back_inserter(res));
    }
    return res;
}
