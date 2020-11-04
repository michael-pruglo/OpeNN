#include "pch.h"
#include "../OpeNN/utility.hpp"

namespace openn
{
	TEST(RandGeneratorTest, RanddRange01)
	{
		for (int i = 0; i < 100; ++i)
		{
			const auto r = randd();
			const double EPS = 1e-9;
			ASSERT_GE(r, 0 -EPS);
			ASSERT_LE(r, 1 +EPS);
		}
	}
}