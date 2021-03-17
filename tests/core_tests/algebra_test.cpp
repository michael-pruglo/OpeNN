#include <tests/common/helpers.hpp>
#include <core/algebra.hpp>

namespace openn::algebra
{
    TEST(CoreAlgebraTest, norm_diff)
    {

    }

    TEST(CoreAlgebraTest, matrix_mul_vec)
    {

    }

    TEST(CoreAlgebraTest, vec_plus_vec)
    {
        using core::operator+;
        EXPECT_EQ(core::Vec({ 1., 2., 3. }) + core::Vec({ 6., 17., 8. }), core::Vec({ 7., 19., 11. }));
    }
}