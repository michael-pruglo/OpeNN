#include <tests/common/helpers.hpp>
#include <core/algebra.hpp>

namespace openn::algebra
{
    core::float_t rand_float_t()
    {
        return rand()%9001/1000.0;
    }

    TEST(CoreAlgebraDeathTest, NormDiff)
    {

    }

    TEST(CoreAlgebraDeathTest, MatrixMulVecSmall)
    {

    }

    TEST(CoreAlgebraDeathTest, VecPlusVecSmall)
    {
        using core::operator+;
        expect_double_vec_eq(core::Vec{}+core::Vec{}, core::Vec{});
        expect_double_vec_eq(core::Vec{1.7}+core::Vec{3.7}, core::Vec{5.4});
    }
    
    TEST(CoreAlgebraDeathTest, VecPlusVecDeath)
    {
        using core::operator+;
        EXPECT_DEATH(core::Vec{}+core::Vec({1}), "");
        EXPECT_DEATH(core::Vec({1,2,3})+core::Vec({1}), "");
        EXPECT_DEATH(core::Vec({1,2})+core::Vec({1,2,3}), "");
    }

    TEST(CoreAlgebraDeathTest, VecPlusVecRand)
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