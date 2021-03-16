#pragma once

#include <tests/common/helpers.hpp>
#include <unordered_map>
#include <functional>

namespace test_core
{
    template<typename T>
    class Database
    {
    public:
        using ParamType_t = int;
        using Data_t = std::unordered_map<ParamType_t, std::vector<T>>;
        using Filter_f = std::function<bool(ParamType_t)>;

    public:
        Database(Data_t data_) : data(std::move(data_)) { }
        virtual ~Database() = default;

        [[nodiscard]] std::vector<T> get_everything() const;
        [[nodiscard]] std::vector<T> get_by_type(ParamType_t type) const;
        [[nodiscard]] std::vector<T> get_by_filter(Filter_f pred) const;

    protected:
        Data_t data;
    };
}

#include "ParamDB_impl.hpp"