#include <tests/common/helpers.hpp>
#include <core/random.hpp>


#include <iostream>


namespace openn::random
{
    using core::operator<<;

    TEST(CoreRandomTest, Seed)
    {
        //const auto& prev_run_initial = core::generate(10, core::rand_d);
        //std::cout << prev_run_initial;
    }

    template<typename InputIterator>
    double diversity(InputIterator first, InputIterator last)
    {
        double sum = 0.0, prev = *first;
        const auto& n = std::distance(first, last);
        while (++first != last)
        {
            sum += *first - prev;
            prev = *first;
        }
        return sum / n;
    }

    void test_interval_d(double l, double r)
    {
        constexpr const int SAMPLE_SIZE = 1'000'000, ZONES = 100;
        const auto& zone_idx = [r,l](double val) -> size_t {
            const auto& zone_width = (r-l)/ZONES;
            const size_t idx = std::floor((val - l) / zone_width);
            return idx;
        };

        double max_generated = l, min_generated = r;
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