#include <tests/common/helpers.hpp>
#include <core/types.hpp>

namespace core::types
{
    void test_dimensions(const core::Matrix& m, size_t exp_rows, size_t exp_cols)
    {
        EXPECT_EQ(m.rows(), exp_rows);
        EXPECT_EQ(m.cols(), exp_cols);
    }

    TEST(CoreTypesTest, Ctors)
    {
        test_dimensions(core::Matrix(1, { 1. }), 1, 1);
        test_dimensions(core::Matrix(7, { 1. }), 7, 1);
        test_dimensions(core::Matrix(43, std::vector<core::float_t>(17, 8.9)), 43, 17);
    }

    TEST(CoreTypesTest, Getters)
    {
        test_dimensions({}, 0, 0);
        test_dimensions({{1.}}, 1, 1);
        test_dimensions({{1.,2.,3.,4.}}, 1, 4);
        test_dimensions({{1.},{2.},{3.},{4.}}, 4, 1);
        test_dimensions({
            { 1.,2.,3.,4.,5.,6.,7. },
            { 2.,2.,3.,4.,5.,6.,7. },
            { 3.,2.,3.,4.,5.,6.,7. },
            { 4.,2.,3.,4.,5.,6.,7. },
            { 5.,2.,3.,4.,5.,6.,7. },
        }, 5, 7);
    }
}
