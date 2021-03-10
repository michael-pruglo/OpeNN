#include <TestOpenn/helpers.hpp>
#include <TestOpenn/ParamDB.hpp>
#include <Core/utility.hpp>

namespace openn
{
	using Param = std::vector<int>;

	class CoreUtilityFixture : public testing::TestWithParam<Param> {};
	TEST_P(CoreUtilityFixture, VecOutput)
	{
		using core::operator<<;
		
		std::ostringstream ss;
		std::vector<int> vi = {5, 6, 7, 8};
		ss << vi;
		EXPECT_EQ(ss.str(), "[ 5, 6, 7, 8 ]");
	}

	enum ParamType : test_core::ParamDB<Param>::ParamType_t
	{
		REGULAR,
		CORNER_CASES,
	};

	test_core::ParamDB<Param> db({
		{
			REGULAR,
			{
				{ 1, 2, 3, 4 },
			}
		},
	});

	INSTANTIATE_TEST_CASE_P(Standard, CoreUtilityFixture, );
}