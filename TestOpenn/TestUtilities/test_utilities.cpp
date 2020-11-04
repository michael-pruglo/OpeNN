#include "../helpers.hpp"
#include "../../OpeNN/utility.hpp"

namespace openn
{
	TEST(RandGeneratorTest, RanddRange01)
	{
		for (int i = 0; i < 100; ++i)
			AssertInRange(randd());
	}
}