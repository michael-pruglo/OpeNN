#include <tests/common/helpers.hpp>
#include <core/algebra.hpp>

namespace core::algebra
{
    core::float_t rand_float_t()
    {
        return rand()%9001/1000.0;
    }

    namespace activation_f
    {
        TEST(CoreAlgebraDeathTest, ActivationSigmoid)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::sigmoid(x), exp, 1e-11);
            };
            tst(-20.00, 0.00000000206);
            tst(-15.39, 0.00000020711);
            tst( -5.67, 0.00343601835);
            tst( -2.06, 0.11304583007);
            tst( -0.53, 0.37051688803);
            tst(  0.00, 0.5);
            tst(  0.44, 0.60825903075);
            tst(  2.11, 0.89187133324);
            tst(  6.55, 0.99857192671);
            tst( 14.61, 0.99999954819);
            tst( 20.00, 0.99999999794);
        }

        TEST(CoreAlgebraDeathTest, ActivationReLU)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(core::relu(x), exp);
            };
            tst(-20.00, 0.);
            tst( -4.00, 0.);
            tst(  0.00, 0.);
            tst(  3.14, 3.14);
            tst( 17.00, 17.);
        }

        TEST(CoreAlgebraDeathTest, ActivationSoftplus)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::softplus(x), exp, 1e-7);
            };
            tst( 1.0, 1.31326163);
            tst(-0.5, 0.474076986);
            tst( 3.4, 3.43282847042);
            tst(-2.1, 0.115519524);
            tst( 0.0, 0.693147182);
            tst(-6.5, 0.00150233845);
        }

        TEST(CoreAlgebraDeathTest, ActivationTanh)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::tanh(x), exp, 1e-11);
            };
            tst(-7.80, -0.999999664235);
            tst(-0.75, -0.635148952387);
            tst( 0.00, 0.);
            tst( 1.00, 0.761594155956);
            tst( 3.14, 0.996260204946);
        }
    }

    namespace derivative_f
    {
        TEST(CoreAlgebraDeathTest, DerivativeSigmoid)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::der_sigmoid(x), exp, 1e-8);
            };
            tst(-5.0, .0066480567);
            tst(-1.4, .1586849);
            tst(-0.3, .24445831);
            tst(0.0, 0.25);
            tst(.6, .22878424);
            tst(1.7, .13060575);
            tst(19.0, 5.602796e-9);
        }

        TEST(CoreAlgebraDeathTest, DerivativeReLU)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_DOUBLE_EQ(core::der_relu(x), exp);
            };
            tst(-17.45, 0.);
            tst(-2.5, 0.);
            tst(-1e-11, 0.);
            tst(0., 1.);
            tst(1e-11, 1.);
            tst(10., 1.);
            tst(182., 1.);
        }

        TEST(CoreAlgebraDeathTest, DerivativeSoftplus)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::der_softplus(x), exp, 1e-11);
            };
            tst(-20.00, 0.00000000206);
            tst(-15.39, 0.00000020711);
            tst( -5.67, 0.00343601835);
            tst( -2.06, 0.11304583007);
            tst( -0.53, 0.37051688803);
            tst(  0.00, 0.5);
            tst(  0.44, 0.60825903075);
            tst(  2.11, 0.89187133324);
            tst(  6.55, 0.99857192671);
            tst( 14.61, 0.99999954819);
            tst( 20.00, 0.99999999794);
        }

        TEST(CoreAlgebraDeathTest, DerivativeTanh)
        {
            const auto tst = [](float_t x, float_t exp){
                EXPECT_NEAR(core::der_tanh(x), exp, 1e-8);
            };
            tst(-5.0, 1.815832e-4);
            tst(-1.4, 0.21615246);
            tst(-0.3, 0.91513696);
            tst(0.0, 1.0);
            tst(.6, .71157776);
            tst(1.7, .12500987);
            tst(19.0, 2.220446e-16);
        }
    }

    namespace cost_f
    {
        TEST(CoreAlgebraDeathTest, MSESmall)
        {
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{}, core::Vec{}), 0.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{7.}, core::Vec{7.}), 0.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{2., 5., 3.}, core::Vec{1., 7., 0.}), 14.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{-2., -5., -3.}, core::Vec{-1., -7., 0.}), 14.);
            EXPECT_DOUBLE_EQ(core::mean_squared_eror(core::Vec{-5., 0., -1., 7.}, core::Vec{-4., 3., -5., 6.}), 27.);
        }

        TEST(CoreAlgebraDeathTest, MSEDeath)
        {
            // the vectors must have equal length
            EXPECT_DEATH(core::mean_squared_eror(core::Vec{}, core::Vec{1.}), "");
            EXPECT_DEATH(core::mean_squared_eror(core::Vec{1., 2., 3.}, core::Vec{1.}), "");
            EXPECT_DEATH(core::mean_squared_eror(core::Vec{1., 2.}, core::Vec{1., 2., 3.}), "");
        }

        TEST(CoreAlgebraDeathTest, CrossEntropySmall)
        {
            EXPECT_DOUBLE_EQ(core::cross_entropy(core::Vec{}, core::Vec{}), 0.);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.1}), 0.32508297, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.2}), 0.54480543, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.3}), 0.76452789, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.4}), 0.98425035, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{0.9}), 2.08286264, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{1.9}), 4.28008721, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1}, core::Vec{-0.1}), -0.11436194, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{0.1,0.999}, core::Vec{0.,1.}), 0.1063610203, 1e-8);
            EXPECT_NEAR(core::cross_entropy(core::Vec{.001,.5,.7,.9,.75}, core::Vec{0.,.5,.3,.2,44.}), -43.44557503, 1e-8);
        }

        TEST(CoreAlgebraDeathTest, CrossEntropyDeath)
        {
            // the vectors must have equal length
            EXPECT_DEATH(core::cross_entropy(core::Vec{}, core::Vec{1.}), "");
            EXPECT_DEATH(core::cross_entropy(core::Vec{1., 2., 3.}, core::Vec{1.}), "");
            EXPECT_DEATH(core::cross_entropy(core::Vec{1., 2.}, core::Vec{1., 2., 3.}), "");
        }
    }

    namespace matrix_vector
    {
        TEST(CoreAlgebraDeathTest, MatrixMulVecSmall)
        {
            using core::operator*;
            expect_double_vec_eq((core::Matrix{}*core::Vec{}), core::Vec{});
            expect_double_vec_eq((core::Matrix{{1.,-1.,2.},{0.,-3.,1.}}*core::Vec{2.,1.,0.}), (core::Vec{1.,-3.}));
            expect_double_vec_eq((core::Matrix{{1.,2.,3.},{4.,5.,6.},{7.,8.,9.},{10.,11.,12.}}*core::Vec{-2.,1.,0.}), core::Vec{0.,-3.,-6.,-9.});
            expect_double_vec_eq((core::Matrix{{1.,2.,3.},{4.,5.,6.},{7.,8.,9.}}*core::Vec{2.,1.,3.}), core::Vec{13.,31.,49.});
            expect_double_vec_eq((core::Matrix{{2.,3.},{4.,5.},{-1.,6.}}*core::Vec{4.,7.}), (core::Vec{29.,51.,38.}));
        }

        TEST(CoreAlgebraDeathTest, MatrixMulVecDeath)
        {
            // matrix with M columns can only be multiplied by a vector of length M
            using core::operator*;
            EXPECT_DEATH((core::Matrix{{1.}}*core::Vec{}), "");
            EXPECT_DEATH((core::Matrix{{1.,2.,3.}}*core::Vec{1.,2.}), "");
            EXPECT_DEATH((core::Matrix{{1.,2.,3.},{1.,2.,3.}}*core::Vec{1.,2.}), "");
            EXPECT_DEATH((core::Matrix{{1.,2.},{1.,2.},{1.,2.}}*core::Vec{1.,2.,3.}), "");
        }

        TEST(CoreAlgebraDeathTest, VecPlusVecSmall)
        {
            using core::operator+;
            expect_double_vec_eq(core::Vec{}+core::Vec{}, core::Vec{});
            expect_double_vec_eq(core::Vec{1.7}+core::Vec{3.7}, core::Vec{5.4});
        }

        TEST(CoreAlgebraDeathTest, VecPlusVecDeath)
        {
            // the vectors must have equal length
            using core::operator+;
            EXPECT_DEATH((core::Vec{}+core::Vec{1.}), "");
            EXPECT_DEATH((core::Vec{1.,2.,3.}+core::Vec{1.}), "");
            EXPECT_DEATH((core::Vec{1.,2.}+core::Vec{1.,2.,3.}), "");
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

        TEST(CoreAlgebraDeathTest, HadamardSmall)
        {
            expect_double_vec_eq(core::hadamard(core::Vec{}, core::Vec{}), core::Vec{});
            expect_double_vec_eq(core::hadamard(core::Vec{1.7}, core::Vec{3.7}), core::Vec{6.29});
            expect_double_vec_eq(core::hadamard(core::Vec{6., 14., 0., -9., -3.}, core::Vec{-8., 5., 0., 0., -8.,}), core::Vec{-48., 70., 0., 0., 24.});
        }

        TEST(CoreAlgebraDeathTest, HadamardDeath)
        {
            // the vectors must have equal length
            using core::operator+;
            EXPECT_DEATH(core::hadamard(core::Vec{}, core::Vec{1.}), "");
            EXPECT_DEATH(core::hadamard(core::Vec{1.,2.,3.}, core::Vec{1.}), "");
            EXPECT_DEATH(core::hadamard(core::Vec{1.,2.}, core::Vec{1.,2.,3.}), "");
        }

        TEST(CoreAlgebraDeathTest, MatrixTranspose)
        {
            expect_double_matrix_eq((core::Matrix{}.t()), (core::Matrix{}));
            expect_double_matrix_eq((core::Matrix{{1.}}.t()), (core::Matrix{{1.}}));
            expect_double_matrix_eq((core::Matrix{{1.}, {2.}}.t()), (core::Matrix{{1., 2.}}));
            expect_double_matrix_eq((core::Matrix{{1., 2.}}.t()), (core::Matrix{{1.}, {2.}}));
            expect_double_matrix_eq((core::Matrix{{1.,2.,3.},{4.,5.,6.},{7.,8.,9.},{10.,11.,12.},{13.,14.,15.}}.t()), (core::Matrix{{1.,4.,7.,10.,13.}, {2.,5.,8.,11.,14.,}, {3.,6.,9.,12.,15.}}));
        }
    }
}
