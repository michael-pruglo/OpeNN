#include "test_nn_param_database.hpp"

using namespace openn;
using Ins = ConstructNNParam::Insertion;

std::unordered_map<ParamDatabase::ParamType, std::vector<ConstructNNParam>> ParamDatabase::database = {
	{
		ParamType::NO_INSERTIONS,
		{
			ConstructNNParam{ 0, 0 },
			ConstructNNParam{ 0, 1 },
			ConstructNNParam{ 1, 0 },
			ConstructNNParam{ 1, 1 },
			ConstructNNParam{ 1, 100 },
			ConstructNNParam{ 100, 1 },
			ConstructNNParam{ 5, 1 },
			ConstructNNParam{ 1, 5 },
			ConstructNNParam{ 100, 100 },
		}
	},

	{
		ParamType::INS,
		{
			ConstructNNParam{ 2, 2, { Ins(7) } },
			ConstructNNParam{ 2, 3, { Ins(1) } },
			ConstructNNParam{ 7, 7, { Ins(1), Ins(12) } },
			ConstructNNParam{ 7, 7, { Ins(11), Ins(1), Ins(3) } }
		}
	},

	{
		ParamType::INS_CORNER_CASES,
		{
			ConstructNNParam{ 0, 0, { Ins(7) } },
			ConstructNNParam{ 0, 3, { Ins(1) } },
			ConstructNNParam{ 2, 0, { Ins(2) } },
			ConstructNNParam{ 2, 3, { Ins(0) } },
			ConstructNNParam{ 0, 0, { Ins(0) } },
			ConstructNNParam{ 0, 0, { Ins(2), Ins(0), Ins(1), Ins(0) } },
			ConstructNNParam{ 0, 0, { Ins(0), Ins(0), Ins(0), Ins(0) } }
		}
	},

	{
		ParamType::INS_AT,
		{
			ConstructNNParam{ 7, 8, { Ins(4, 0) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1), Ins(4, 1) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1), Ins(4, 2) } },
			ConstructNNParam{ 6, 9, { Ins(7, 1), Ins(4, 1), Ins(6, 1) } }
		}
	},

	{
		ParamType::INS_AT_CORNER_CASES,
		{
			ConstructNNParam{ 0, 0, { Ins(7, 0) } },
			ConstructNNParam{ 0, 3, { Ins(1, 1) } },
			ConstructNNParam{ 2, 0, { Ins(2, 0) } },
			ConstructNNParam{ 2, 3, { Ins(0, 1) } },
			ConstructNNParam{ 0, 0, { Ins(0, 1) } },
			ConstructNNParam{ 0, 0, { Ins(2, 0), Ins(0, 2), Ins(1, 1), Ins(0, 0) } },
			ConstructNNParam{ 0, 0, { Ins(0, 0), Ins(0, 2), Ins(0, 1), Ins(0, 0) } },
			ConstructNNParam{ 1, 1, { Ins(1, 1), Ins(1, 2), Ins(1, 3), Ins(1, 4), Ins(1, 5) } },
			ConstructNNParam{ 1, 1, { Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0) } }
		}
	},

	{
		ParamType::STRESS_TESTS,
		{
			ConstructNNParam{ 10000, 10000, { Ins(7) } },
			ConstructNNParam{ 2, 2, std::vector<Ins>(20, Ins(1000)) },
			ConstructNNParam{ 1000, 1000, std::vector<Ins>(20, Ins(1000)) },
			ConstructNNParam{ 1000, 1000, std::vector<Ins>(20, Ins(1000, 0)) },
		}
	}
};
