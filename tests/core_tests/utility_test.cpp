#include <GTest/TestCommon/helpers.hpp>
#include <GTest/TestCommon/ParamDB.hpp>
#include <GTest/TestCommon/ParamDB.cpp>
#include <OpeNN/Core/utility.hpp>

namespace openn
{
    using TestCase = std::pair<std::vector<int>, std::string>;

    class CoreUtilityFixture : public testing::TestWithParam<TestCase> {};
    TEST_P(CoreUtilityFixture, VecOutput)
{
    using core::operator<<;
    const auto& [vec, expected] = GetParam();

    std::ostringstream ss;
    ss << vec;
    EXPECT_EQ(ss.str(), expected);
}

enum ParamType : test_core::Database<TestCase>::ParamType_t
{
    REGULAR,
    CORNER_CASES,
};

test_core::Database<TestCase> db({
                                         {
                                                 REGULAR,
                                                 {
                                                         { { 1, 2, 3, 4 }, "[ 1, 2, 3, 4 ]" },
                                                         { { 9 }, "[ 9 ]" },
                                                 }
                                         },
                                         {
                                                 CORNER_CASES,
                                                 {
                                                         { { }, "[  ]" },
                                                 }
                                         },
                                 });

INSTANTIATE_TEST_CASE_P(Standard, CoreUtilityFixture, testing::ValuesIn(db.get_everything()));
}