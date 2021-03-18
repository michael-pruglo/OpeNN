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

    core::float_t rand_float_t()
    {
        return rand()%9001/1000.0;
    }

    TEST(CoreAlgebraTest, vec_plus_vec)
    {
        using core::operator+;
        constexpr int TEST_CASES_AMOUNT = 1'000, MAX_VEC_LEN = 1'000;

        srand(time(nullptr));
        for (int test = 0; test < TEST_CASES_AMOUNT; ++test)
        {
            core::Vec v1, v2, expected;
            const int LEN = rand()%MAX_VEC_LEN;
            v1.reserve(LEN);
            v2.reserve(LEN);
            expected.reserve(LEN);
            for (int i = 0; i < LEN; ++i)
            {
                core::float_t a = rand_float_t(), b = rand_float_t();
                v1.push_back(a);
                v2.push_back(b);
                expected.push_back(a+b);
            }

            expect_double_vec_eq(v1+v2, expected);
        }
    }
}