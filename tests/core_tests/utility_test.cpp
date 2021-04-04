#include <tests/common/helpers.hpp>
#include <core/utility.hpp>
#include <queue>
#include <array>

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
    /*
    namespace generate
    {
        TEST(CoreUtilityTest, Generate)
        {
            const auto& gen_42 = [](){ return 42; };
            const auto& gen_str = [](){ return std::string("str"); };
            const auto& gen_rec = [](){ return std::vector<int>{ 1 }; };
            int siota = 0;
            const auto& gen_iota = [&siota](){ return siota++; };

            EXPECT_EQ(core::generate(0, gen_42), std::vector<int>{ });
            EXPECT_EQ(core::generate(1, gen_42), std::vector<int>(1, 42));
            EXPECT_EQ(core::generate(17, gen_42), std::vector<int>(17, 42));

            EXPECT_EQ(core::generate(0, gen_str), std::vector<std::string>{ });
            EXPECT_EQ(core::generate(1, gen_str), std::vector<std::string>(1, "str"));
            EXPECT_EQ(core::generate(17, gen_str), std::vector<std::string>(17, "str"));

            EXPECT_EQ(core::generate(0, gen_rec), std::vector<std::vector<int>>{ });
            EXPECT_EQ(core::generate(1, gen_rec), std::vector<std::vector<int>>(1, std::vector<int>{1}));
            EXPECT_EQ(core::generate(17, gen_rec), std::vector<std::vector<int>>(17, std::vector<int>{1}));

            siota = 0; EXPECT_EQ(core::generate(0, gen_iota), std::vector<int>({ }));
            siota = 0; EXPECT_EQ(core::generate(1, gen_iota), std::vector<int>({ 0 }));
            siota = 0; EXPECT_EQ(core::generate(3, gen_iota), std::vector<int>({ 0,1,2 }));
            siota = 0; EXPECT_EQ(core::generate(17, gen_iota), std::vector<int>({ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 }));

            EXPECT_EQ(core::generate(0, [](){ return 42; }), std::vector<int>{ });
            EXPECT_EQ(core::generate(1, [](){ return 42; }), std::vector<int>(1, 42));
            EXPECT_EQ(core::generate(17, [](){ return 42; }), std::vector<int>(17, 42));
        }

        TEST(CoreUtilityTest, GenerateI)
        {
            const auto& gen_42 = [](){ return 42; };
            const auto& gen_const = [](size_t i){ return 42; };
            const auto& gen_rev = [](size_t i){ return 10-static_cast<int>(i); };
            const auto& gen_iota = [](size_t i){ return static_cast<int>(i); };
            const auto& gen_fun = [](size_t i){ return static_cast<int>(3*i*i*i + 27*i*i - 34*i - 2); };
            const auto& gen_dbl = [](size_t i){ return i/2.; };

            EXPECT_EQ(core::generate_i(0, gen_const), std::vector<int>({}));
            EXPECT_EQ(core::generate_i(1, gen_const), std::vector<int>({ 42 }));
            EXPECT_EQ(core::generate_i(13, gen_const), std::vector<int>(13, 42));

            EXPECT_EQ(core::generate_i(75, gen_const), core::generate(75, gen_42));

            EXPECT_EQ(core::generate_i(13, gen_rev), std::vector<int>({ 10,9,8,7,6,5,4,3,2,1,0,-1,-2 }));

            EXPECT_EQ(core::generate_i(13, gen_iota), std::vector<int>({ 0,1,2,3,4,5,6,7,8,9,10,11,12 }));

            EXPECT_EQ(core::generate_i(6, gen_fun), std::vector<int>({ -2,-6,62,220,486,878 }));

            EXPECT_EQ(core::generate_i(6, [](size_t i){ return static_cast<int>(3*i*i*i + 27*i*i - 34*i - 2); }), std::vector<int>({ -2,-6,62,220,486,878 }));

            EXPECT_EQ(core::generate_i(6, gen_dbl), std::vector<double>({ 0., 0.5, 1., 1.5, 2., 2.5 }));
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
            EXPECT_EQ(core::map(gen_inc, templ_iota), core::generate_i(5, gen_inc));

            EXPECT_EQ(core::map(my_abs<int>, templ_rev), std::vector<int>({ 2,1,0,1,2,3 }));
            EXPECT_EQ(core::map(std::labs, templ_rev), std::vector<long>({ 2,1,0,1,2,3 }));

            EXPECT_EQ(core::map([](int i){ return i-41; }, templ_const), std::vector<int>(5, 1));

            expect_double_vec_eq(core::map([](double d){ return d/3.; }, templ_fl), { 6.1, 5.2, 6.4 });

            EXPECT_EQ(core::map([](double _){ return 42.; }, std::vector<double>{}), std::vector<double>{});
            EXPECT_EQ(core::map([](std::string _){ return std::string("str"); }, std::vector<std::string>{}), std::vector<std::string>{});
        }
    }
    */
}
