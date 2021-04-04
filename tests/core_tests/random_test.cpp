#include <tests/common/helpers.hpp>
#include <core/random.hpp>
#include <algorithm>
#include <array>

namespace core::random
{
    using core::operator<<;
    using core::float_t;

    TEST(CoreRandomTest, Seed)
    {
        const std::vector<float_t> prev_run_initial{ 0.271145, 0.439242, 0.105885, 0.31747, 0.639287, 0.232686, 0.589953, 0.382386, 0.326701, 0.0690268 };
        std::vector<float_t> new_run_initial(10);
        std::generate_n(new_run_initial.begin(), 10, []{ return core::rand_d(); });
        
        ASSERT_NE(prev_run_initial, new_run_initial);
    }

    template<typename InputIterator>
    float_t diversity(InputIterator first, InputIterator last)
    {
        float_t sum = 0.0, prev = *first;
        const auto& n = std::distance(first, last);
        while (++first != last)
        {
            sum += *first - prev;
            prev = *first;
        }
        return sum / n;
    }

    constexpr const int SAMPLE_SIZE = 100'000, ZONES = 100;
    size_t zone_idx(float_t l, float_t r, float_t val)
    {
        const float_t zone_width = (r-l) / ZONES;
        return zone_width > 1e-7 ? std::floor((val-l) / zone_width) : 0.;
    };

    template<typename RNGen>
    void test_interval(auto l, auto r, RNGen random_gen)
    {
        auto max_generated = l, min_generated = r;
        std::array<int, ZONES+1> amount_by_zone{};
        amount_by_zone.fill(0);

        for (int i = 0; i < SAMPLE_SIZE; ++i)
        {
            const auto gen = random_gen(l, r);
            EXPECT_GE(gen, l);
            EXPECT_LE(gen, r);

            min_generated = std::min(gen, min_generated);
            max_generated = std::max(gen, max_generated);

            const auto idx = zone_idx(l, r, gen);
            ASSERT_LT(idx, amount_by_zone.size());
            ++amount_by_zone.at(idx);
        }
        const auto tolerance = (r-l) * 5. / 100.;
        EXPECT_NEAR(min_generated, l, tolerance);
        EXPECT_NEAR(max_generated, r, tolerance);

        std::ostringstream ss;
        ss << amount_by_zone;
        EXPECT_LE(diversity(amount_by_zone.begin(), amount_by_zone.end()), .05 * SAMPLE_SIZE/ZONES) << ss.str();
    }

    TEST(CoreRandomTest, RandD)
    {
        test_interval(0.0, 1.0, core::rand_d<float_t>);
        test_interval(-1.0, 1.0, core::rand_d<float_t>);
        test_interval(1.76, 1.76, core::rand_d<float_t>);
        test_interval(0.0, 100.0, core::rand_d<float_t>);
        test_interval(95.0, 178.99, core::rand_d<float_t>);
        test_interval(-17.0, -4.34, core::rand_d<float_t>);
    }

    TEST(CoreRandomTest, RandI)
    {
        test_interval(0, 10, core::rand_i<int>);
        test_interval(-10, 10, core::rand_i<int>);
        test_interval(-64, 60, core::rand_i<int>);
        test_interval(-64, -13, core::rand_i<int>);
        test_interval(14, 14, core::rand_i<int>);
        test_interval(-1234567, 12345678, core::rand_i<int>);
    }

    TEST(CoreRandomTest, RandVec)
    {
        for (size_t sz: { 0, 1, 2, 5, 9, 42, 178, 599, 1234567})
            ASSERT_EQ(core::rand_vec(sz).size(), sz);
    }

    TEST(CoreRandomTest, RandMatrix)
    {
        for (size_t n: { 0, 1, 2, 5, 9, 42})
            for (size_t m: { 0, 1, 2, 5, 9, 42})
            {
                const core::Matrix& mat = core::rand_matrix(n, m);
                ASSERT_EQ(mat.size(), n);
                for (const auto& row: mat)
                    ASSERT_EQ(row.size(), m);
            }
    }
}