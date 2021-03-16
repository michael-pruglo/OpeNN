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
