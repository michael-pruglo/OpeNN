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
            EXPECT_TRUE(core::is_equal(123456789.0123, 123456789.0123, 1.));
            EXPECT_TRUE(core::is_equal(123457789.0123, 123456789.0123, 1000.));
            EXPECT_FALSE(core::is_equal(123457789.0123, 123456789.0123, 999.));
            EXPECT_TRUE(core::is_equal(1e-9, 1e-9+1e-10, 1e-9));
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

    }
}
