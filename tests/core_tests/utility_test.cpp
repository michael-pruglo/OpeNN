#include <tests/common/helpers.hpp>
#include <core/utility.hpp>

namespace openn::utility
{
    namespace vec_output
    {
        struct Printable42{};
        void operator<<(std::ostream& os, const Printable42& _){ os<<"42"; }

        TEST(CoreUtilityTest, VecOutput_Empty)
        {
            using core::operator<<;

            { std::ostringstream ss; ss << std::vector<int>{};              EXPECT_EQ(ss.str(), "[  ]"); }
            { std::ostringstream ss; ss << std::vector<float>{};            EXPECT_EQ(ss.str(), "[  ]"); }
            { std::ostringstream ss; ss << std::vector<std::string>{};      EXPECT_EQ(ss.str(), "[  ]"); }
            { std::ostringstream ss; ss << std::vector<std::vector<int>>{}; EXPECT_EQ(ss.str(), "[  ]"); }
            { std::ostringstream ss; ss << std::vector<Printable42>{};      EXPECT_EQ(ss.str(), "[  ]"); }
        }

        TEST(CoreUtilityTest, VecOutput_Int)
        {
            using core::operator<<;

            { std::ostringstream ss; ss << std::vector<int>{ 1 };           EXPECT_EQ(ss.str(), "[ 1 ]"); }
            { std::ostringstream ss; ss << std::vector<int>{ -91 };         EXPECT_EQ(ss.str(), "[ -91 ]"); }
            { std::ostringstream ss; ss << std::vector<int>{ 0 };           EXPECT_EQ(ss.str(), "[ 0 ]"); }
            { std::ostringstream ss; ss << std::vector<int>{ 15, 87 };      EXPECT_EQ(ss.str(), "[ 15, 87 ]"); }
            { std::ostringstream ss; ss << std::vector<int>{ -14, 2 };      EXPECT_EQ(ss.str(), "[ -14, 2 ]"); }
            { std::ostringstream ss; ss << std::vector<int>{ -5, 0, 5 };    EXPECT_EQ(ss.str(), "[ -5, 0, 5 ]"); }
            { std::ostringstream ss; ss << std::vector<int>{ -51, 1234224828, 0, -89234781 }; EXPECT_EQ(ss.str(), "[ -51, 1234224828, 0, -89234781 ]"); }
        }

        TEST(CoreUtilityTest, VecOutput_Floating)
        {
            using core::operator<<;

            { std::ostringstream ss; ss << std::vector<double>{ 1. };                   EXPECT_EQ(ss.str(), "[ 1 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ -91.7842144 };          EXPECT_EQ(ss.str(), "[ -91.7842 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ 91.7842144 };           EXPECT_EQ(ss.str(), "[ 91.7842 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ -12345.678901234 };     EXPECT_EQ(ss.str(), "[ -12345.7 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ -12345678.901234 };     EXPECT_EQ(ss.str(), "[ -1.23457e+07 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ -91.000001 };           EXPECT_EQ(ss.str(), "[ -91 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ 0. };                   EXPECT_EQ(ss.str(), "[ 0 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ 0.000007 };             EXPECT_EQ(ss.str(), "[ 7e-06 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ 15., 87. };             EXPECT_EQ(ss.str(), "[ 15, 87 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ -14.114123413, 2.431 }; EXPECT_EQ(ss.str(), "[ -14.1141, 2.431 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ 123456789.11, 0.000000091, 5.470000000 }; EXPECT_EQ(ss.str(), "[ 1.23457e+08, 9.1e-08, 5.47 ]"); }
            { std::ostringstream ss; ss << std::vector<double>{ -51., 1234224828., 0., -89234781. }; EXPECT_EQ(ss.str(), "[ -51, 1.23422e+09, 0, -8.92348e+07 ]"); }
        }

        TEST(CoreUtilityTest, VecOutput_Recursive)
        {
            using core::operator<<;

            { std::ostringstream ss; ss << std::vector<std::vector<int>>{ {1,-2},{} };  EXPECT_EQ(ss.str(), "[ [ 1, -2 ], [  ] ]"); }
            { std::ostringstream ss; ss << std::vector<std::vector<int>>{ {1,2},{-9,1283} };  EXPECT_EQ(ss.str(), "[ [ 1, 2 ], [ -9, 1283 ] ]"); }
            { std::ostringstream ss; ss << std::vector<std::vector<int>>{ {1,4},{1,0},{-1,9},{-9,1283} };  EXPECT_EQ(ss.str(), "[ [ 1, 4 ], [ 1, 0 ], [ -1, 9 ], [ -9, 1283 ] ]"); }
        }

        TEST(CoreUtilityTest, VecOutput_Struct)
        {
            using core::operator<<;

            { std::ostringstream ss; ss << std::vector<Printable42>(1); EXPECT_EQ(ss.str(), "[ 42 ]"); }
            { std::ostringstream ss; ss << std::vector<Printable42>(2); EXPECT_EQ(ss.str(), "[ 42, 42 ]"); }
            { std::ostringstream ss; ss << std::vector<Printable42>(5); EXPECT_EQ(ss.str(), "[ 42, 42, 42, 42, 42 ]"); }
        }
    }

    namespace equal
    {
        TEST(CoreUtilityTest, IsEqual_Integer)
        {
            EXPECT_TRUE(core::is_equal(1, 1));
            EXPECT_TRUE(core::is_equal(0, 0));
            EXPECT_TRUE(core::is_equal(-17, -17));
            EXPECT_TRUE(core::is_equal(43u, 43u));
            EXPECT_TRUE(core::is_equal(74l, 74l));
            EXPECT_TRUE(core::is_equal(-99ll, -99ll));
            EXPECT_TRUE(core::is_equal(99ull, 99ull));
            EXPECT_TRUE(core::is_equal(99/11, 9));
            EXPECT_TRUE(core::is_equal(17%3, 1+1));
            EXPECT_TRUE(core::is_equal(-~7, 8));

            EXPECT_FALSE(core::is_equal(1, 2));
            EXPECT_FALSE(core::is_equal(1u, 2u));
            EXPECT_FALSE(core::is_equal(-8, -7));
        }

        TEST(CoreUtilityTest, IsEqual_Floating)
        {
            EXPECT_TRUE(core::is_equal(18.f, 18.f));
            EXPECT_TRUE(core::is_equal(1.f+2.f, 51.f/17.f));
            EXPECT_TRUE(core::is_equal(1e-9, 1e-9));
            EXPECT_FALSE(core::is_equal(1e-9, 1e-9+1e-10));
            EXPECT_TRUE(core::is_equal(0., -0.));
            EXPECT_TRUE(core::is_equal(1234567890123, 1234567890123));
            EXPECT_FALSE(core::is_equal(1234567890124, 1234567890123));
            EXPECT_TRUE(core::is_equal(76.298L, 76.298L));

            const auto& make_from_parts = [](double d, int n){
                double part = d/n, res = 0.;
                for (int i = 0; i < n; ++i)
                    res += part;
                return res;
            };
            EXPECT_TRUE(core::is_equal(71., make_from_parts(71., 17)));
            EXPECT_TRUE(core::is_equal(1e-7, make_from_parts(1e-7, 19)));
            EXPECT_TRUE(core::is_equal(1e11, make_from_parts(1e11, 19)));
            EXPECT_TRUE(core::is_equal(1e14, make_from_parts(1e14, 19)));
            EXPECT_TRUE(core::is_equal(1e17, make_from_parts(1e17, 19)));
            EXPECT_TRUE(core::is_equal(987654321., make_from_parts(987654321., 2)));
            EXPECT_TRUE(core::is_equal(987654321., make_from_parts(987654321., 5)));
            EXPECT_TRUE(core::is_equal(987654321., make_from_parts(987654321., 7)));
            EXPECT_TRUE(core::is_equal(987654321., make_from_parts(987654321., 9)));

            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 11)));
            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 12)));
            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 13)));
            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 14)));
            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 16)));
            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 32)));
            EXPECT_TRUE(core::is_equal(987., make_from_parts(987., 64)));

        }

        TEST(CoreUtilityTest, IsEqual_FloatingEps)
        {
            EXPECT_TRUE(core::is_equal(16.0001, 16.0007, .001));
            EXPECT_TRUE(core::is_equal(16.002, 16.003, .001));
            EXPECT_TRUE(core::is_equal(123'456'789.0123, 123'456'789.0123, 1.));
            EXPECT_TRUE(core::is_equal(123'457'789.0123, 123'456'789.0123, 1000.));
            EXPECT_FALSE(core::is_equal(123'457'789.0123, 123456789.0123, 999.));
            EXPECT_TRUE(core::is_equal(22'345'678'901'234.0123L, 12'345'678'901'234.0123L, 10'000'000'000'000.L));
        }

        TEST(CoreUtilityTest, IsEqual_Other)
        {
            EXPECT_TRUE(core::is_equal('c', 'c'));
            EXPECT_TRUE(core::is_equal('c', char(99)));
            EXPECT_TRUE(core::is_equal("fluff", "fluff"));
            EXPECT_TRUE(core::is_equal(std::vector<int>{}, std::vector<int>{}));
            struct Comparable { bool operator==(const Comparable& other) const { return true; } };
            EXPECT_TRUE(core::is_equal(Comparable{}, Comparable{}));
            EXPECT_TRUE(core::is_equal(std::vector<Comparable>(6), std::vector<Comparable>(6)));

            EXPECT_FALSE(core::is_equal('t', 'f'));
            EXPECT_FALSE(core::is_equal("fluff", "12345"));
        }
    }

    namespace generate
    {
        TEST(CoreUtilityTest, Generate)
        {
            const auto& gen_42 = [](){ return 42; };
            const auto& gen_str = [](){ return "str"; };
            const auto& gen_rec = [](){ return std::vector<int>{ 1 }; };
            int siota = 0;
            const auto& gen_iota = [&siota](){ return siota++; };

            EXPECT_EQ(core::generate<int>(0, gen_42), std::vector<int>{ });
            EXPECT_EQ(core::generate<int>(1, gen_42), std::vector<int>(1, 42));
            EXPECT_EQ(core::generate<int>(17, gen_42), std::vector<int>(17, 42));

            EXPECT_EQ(core::generate<std::string>(0, gen_str), std::vector<std::string>{ });
            EXPECT_EQ(core::generate<std::string>(1, gen_str), std::vector<std::string>(1, "str"));
            EXPECT_EQ(core::generate<std::string>(17, gen_str), std::vector<std::string>(17, "str"));

            EXPECT_EQ(core::generate<std::vector<int>>(0, gen_rec), std::vector<std::vector<int>>{ });
            EXPECT_EQ(core::generate<std::vector<int>>(1, gen_rec), std::vector<std::vector<int>>(1, std::vector<int>{1}));
            EXPECT_EQ(core::generate<std::vector<int>>(17, gen_rec), std::vector<std::vector<int>>(17, std::vector<int>{1}));

            siota = 0; EXPECT_EQ(core::generate<int>(0, gen_iota), std::vector<int>({ }));
            siota = 0; EXPECT_EQ(core::generate<int>(1, gen_iota), std::vector<int>({ 0 }));
            siota = 0; EXPECT_EQ(core::generate<int>(3, gen_iota), std::vector<int>({ 0,1,2 }));
            siota = 0; EXPECT_EQ(core::generate<int>(17, gen_iota), std::vector<int>({ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 }));

            EXPECT_EQ(core::generate<int>(0, [](){ return 42; }), std::vector<int>{ });
            EXPECT_EQ(core::generate<int>(1, [](){ return 42; }), std::vector<int>(1, 42));
            EXPECT_EQ(core::generate<int>(17, [](){ return 42; }), std::vector<int>(17, 42));
        }

        TEST(CoreUtilityTest, Generate_i)
        {
            const auto& gen_42 = [](){ return 42; };
            const auto& gen_const = [](size_t i){ return 42; };
            const auto& gen_rev = [](size_t i){ return 10-i; };
            const auto& gen_iota = [](size_t i){ return i; };
            const auto& gen_fun = [](size_t i){ return 3*i*i*i + 27*i*i - 34*i - 2; };
            const auto& gen_dbl = [](size_t i){ return i/2.; };

            EXPECT_EQ(core::generate_i<int>(0, gen_const), std::vector<int>({}));
            EXPECT_EQ(core::generate_i<int>(1, gen_const), std::vector<int>({ 42 }));
            EXPECT_EQ(core::generate_i<int>(13, gen_const), std::vector<int>(13, 42));

            EXPECT_EQ(core::generate_i<int>(75, gen_const), core::generate<int>(75, gen_42));

            EXPECT_EQ(core::generate_i<int>(13, gen_rev), std::vector<int>({ 10,9,8,7,6,5,4,3,2,1,0,-1,-2 }));

            EXPECT_EQ(core::generate_i<int>(13, gen_iota), std::vector<int>({ 0,1,2,3,4,5,6,7,8,9,10,11,12 }));

            EXPECT_EQ(core::generate_i<int>(6, gen_fun), std::vector<int>({ -2,-6,62,220,486,878 }));

            EXPECT_EQ(core::generate_i<int>(6, [](size_t i){ return 3*i*i*i + 27*i*i - 34*i - 2; }), std::vector<int>({ -2,-6,62,220,486,878 }));

            EXPECT_EQ(core::generate_i<double>(6, gen_dbl), std::vector<double>({ 0., 0.5, 1., 1.5, 2., 2.5 }));
        }

        template<typename T> T my_abs(T arg) { return arg<0 ? -arg : arg; }

        TEST(CoreUtilityTest, Map)
        {
            const std::vector<int> templ_iota = { 0, 1, 2, 3, 4 };
            const std::vector<int> templ_rev = { 2, 1, 0, -1, -2, -3 };
            const std::vector<int> templ_const = { 42, 42, 42, 42, 42 };
            const std::vector<double> templ_fl = { 18.3, 15.6, 19.2 };

            const auto& gen_inc = [](int i){ return i+1; };

            EXPECT_EQ(core::map([](int i){ return i+1; }, templ_iota), std::vector<int>({ 1,2,3,4,5 }));
            EXPECT_EQ(core::map(gen_inc, templ_iota), std::vector<int>({ 1,2,3,4,5 }));
            EXPECT_EQ(core::map(gen_inc, templ_iota), core::generate_i<int>(5, gen_inc));

            EXPECT_EQ(core::map(my_abs<int>, templ_rev), std::vector<int>({ 2,1,0,1,2,3 }));
            EXPECT_EQ(core::map(std::labs, templ_rev), std::vector<int>({ 2,1,0,1,2,3 }));

            EXPECT_EQ(core::map([](int i){ return i-41; }, templ_const), std::vector<int>(5, 1));

            expect_double_vec_eq(core::map([](double d){ return d/3.; }, templ_fl), { 6.1, 5.2, 6.4 });

            EXPECT_EQ(core::map([](double _){ return 42; }, std::vector<double>{}), std::vector<double>{});
            EXPECT_EQ(core::map([](std::string _){ return "str"; }, std::vector<std::string>{}), std::vector<std::string>{});
        }
    }
}
