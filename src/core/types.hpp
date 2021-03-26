#pragma once

#include <vector>

namespace core
{
    using float_t = double;

    using Vec = std::vector<float_t>;

    using BaseMatrix = std::vector<std::vector<float_t>>;
    class Matrix : public BaseMatrix
    {
    public:
        using BaseMatrix::BaseMatrix;
        Matrix(BaseMatrix v) : BaseMatrix(std::move(v)) {}

        size_t rows() const { return size(); }
        size_t cols() const { return empty() ? 0 : begin()->size(); }
        Matrix t() const;
    };
}
