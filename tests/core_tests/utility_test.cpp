#include <tests/common/helpers.hpp>
#include <core/utility.hpp>

namespace core::utility
{
    namespace equal
    {
        TEST(CoreUtilityTest, IsEqualInteger)
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

        TEST(CoreUtilityTest, IsEqualFloating)
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

        TEST(CoreUtilityTest, IsEqualFloatingEps)
        {
            EXPECT_TRUE(core::is_equal(16.0001, 16.0007, .001));
            EXPECT_TRUE(core::is_equal(16.002, 16.003, .001));
            EXPECT_TRUE(core::is_equal(123'456'789.0123, 123'456'789.0123, 1.));
            EXPECT_TRUE(core::is_equal(123'457'789.0123, 123'456'789.0123, 1000.));
            EXPECT_FALSE(core::is_equal(123'457'789.0123, 123456789.0123, 999.));
            EXPECT_TRUE(core::is_equal(22'345'678'901'234.0123L, 12'345'678'901'234.0123L, 10'000'000'000'000.L));
        }

        TEST(CoreUtilityTest, IsEqualOther)
        {
            EXPECT_TRUE(core::is_equal('c', 'c'));
            EXPECT_TRUE(core::is_equal('c', char(99)));
            EXPECT_TRUE(core::is_equal(std::string("fluff"), std::string("fluff")));
            EXPECT_TRUE(core::is_equal(std::vector<int>{}, std::vector<int>{}));
            struct Comparable { bool operator==(const Comparable& other) const { return true; } };
            EXPECT_TRUE(core::is_equal(Comparable{}, Comparable{}));
            EXPECT_TRUE(core::is_equal(std::vector<Comparable>(6), std::vector<Comparable>(6)));

            EXPECT_FALSE(core::is_equal('t', 'f'));
            EXPECT_FALSE(core::is_equal("address", "address"));
        }
    }
}
