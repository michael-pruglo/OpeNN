#include <tests/common/helpers.hpp>
#include <core/random.hpp>

namespace openn::random
{
    using core::operator<<;
    using core::float_t;

    TEST(CoreRandomTest, Seed)
    {
        const std::vector<float_t>
            prev_run_initial{ 0.271145, 0.439242, 0.105885, 0.31747, 0.639287, 0.232686, 0.589953, 0.382386, 0.326701, 0.0690268 },
            new_run_initial = core::generate(10, []{ return core::rand_d(); });
        
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

    void test_interval_d(float_t l, float_t r)
    {
        constexpr const int SAMPLE_SIZE = 1'000'000, ZONES = 100;
        const auto& zone_idx = [r,l](float_t val) -> size_t {
            const auto& zone_width = (r-l)/ZONES;
            const size_t idx = std::floor((val - l) / zone_width);
            return idx;
        };

        float_t max_generated = l, min_generated = r;
        std::array<int, ZONES> amount_by_zone{};
        for (int i = 0; i < SAMPLE_SIZE; ++i)
        {
            const auto gen = core::rand_d(l, r);

            min_generated = std::min(gen, min_generated);
            max_generated = std::max(gen, max_generated);

            ++amount_by_zone[zone_idx(gen)];
        }
        EXPECT_NEAR(min_generated, l, 1e-3);
        EXPECT_NEAR(max_generated, r, 1e-3);

        std::ostringstream ss;
        ss << amount_by_zone;
        EXPECT_LE(diversity(amount_by_zone.begin(), amount_by_zone.end()), .05 * SAMPLE_SIZE/ZONES) << ss.str();
    }

    TEST(CoreRandomTest, rand_d)
    {
        test_interval_d(0.0, 1.0);
        test_interval_d(-1.0, 1.0);
        test_interval_d(1.76, 1.76);
        test_interval_d(0.0, 100.0);
        test_interval_d(95.0, 178.99);
        test_interval_d(-17.0, -4.34);
    }

}