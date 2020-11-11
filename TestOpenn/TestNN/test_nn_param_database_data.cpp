#include <TestOpenn/TestNN/test_nn_param_database.hpp>
#include <OpeNN/package/opeNN.hpp>

using namespace openn;
using Ins = ConstructNNParam::InsLayer;

std::unordered_map<ParamDatabase::ParamType, std::vector<ConstructNNParam>> ParamDatabase::database = {
	{
		ParamType::NO_INSERTIONS,
		{
			ConstructNNParam( { 0, 0 }, {} ),
			ConstructNNParam{ { 0, 1 }, {} },
			ConstructNNParam{ { 1, 0 }, {} },
			ConstructNNParam{ { 1, 1 }, {} },
			ConstructNNParam{ { 1, 100 }, {} },
			ConstructNNParam{ { 100, 1 }, {} },
			ConstructNNParam{ { 5, 1 }, {} },
			ConstructNNParam{ { 1, 5 }, {} },
			ConstructNNParam{ { 100, 100 }, {} },
		}
	},

	{
		ParamType::NO_INS_MULT,
		{
			ConstructNNParam( { 0, 5, 2, 3, 0 }, {} ),
			ConstructNNParam{ { 0, 7, 0, 0, 1 }, {} },
			ConstructNNParam{ { 0, 0, 0, 0, 0 }, {} },
			ConstructNNParam{ { 1, 0, 0, 0, 0 }, {} },
			ConstructNNParam{ { 1, 1, 1, 9, 1 }, {} },
			ConstructNNParam{ { 100, 100, 100, 100, 100 }, {} },
		}
	},
	
	{
		ParamType::NO_INS_DIFF_ACTIVATIONS,
		{
			ConstructNNParam( { 0, { 5, ActivationFType::softplus }, { 2, ActivationFType::ReLU }, 3, 0 }, {} ),
			ConstructNNParam{ { 0, 7, 0, 0, { 2, ActivationFType::ReLU } }, {} },
			ConstructNNParam{ { 0, 0, { 2, ActivationFType::sigmoid }, 0, { 100, ActivationFType::ReLU } }, {} },
			ConstructNNParam{ { 1, 0, { 0, ActivationFType::ReLU }, 0, 0 }, {} },
			ConstructNNParam{ { { 2, ActivationFType::tanh }, { 2, ActivationFType::ReLU }, { 2, ActivationFType::softplus }, { 2, ActivationFType::ReLU }, { 2, ActivationFType::tanh } }, {} },
		}
	},

	{
		ParamType::INS,
		{
			ConstructNNParam{ { 2, 3 }, { Ins(1) } },
			ConstructNNParam{ { 2, 2 }, { Ins(7) } },
			ConstructNNParam{ { 7, 7 }, { Ins(1), Ins(12) } },
			ConstructNNParam{ { 7, 7 }, { Ins(11), Ins(1), Ins(3) } }
		}
	},

	{
		ParamType::INS_CORNER_CASES,
		{
			ConstructNNParam{ { 0, 0 }, { Ins(7) } },
			ConstructNNParam{ { 0, 3 }, { Ins(1) } },
			ConstructNNParam{ { 2, 0 }, { Ins(2) } },
			ConstructNNParam{ { 2, 3 }, { Ins(0) } },
			ConstructNNParam{ { 0, 0 }, { Ins(0) } },
			ConstructNNParam{ { 0, 0 }, { Ins(2), Ins(0), Ins(1), Ins(0) } },
			ConstructNNParam{ { 0, 0 }, { Ins(0), Ins(0), Ins(0), Ins(0) } }
		}
	},
	
	{
		ParamType::INS_MULT,
		{
			ConstructNNParam{ { 2, 0, 0, 0, 2 }, { Ins(7) } },
			ConstructNNParam{ { 2, 4, 0, 1, 3 }, { Ins(1) } },
			ConstructNNParam{ { 7, 1, 1, 1, 1, 1, 1, 7 }, { Ins(1), Ins(12) } },
			ConstructNNParam{ { 7, 8, 7 }, { Ins(11), Ins(1), Ins(3) } }
		}
	},

	{
		ParamType::INS_AT,
		{
			ConstructNNParam{ { 7, 8 }, { Ins(4, 0) } },
			ConstructNNParam{ { 6, 9 }, { Ins(7, 1) } },
			ConstructNNParam{ { 6, 9 }, { Ins(7, 1), Ins(4, 1) } },
			ConstructNNParam{ { 6, 9 }, { Ins(7, 1), Ins(4, 2) } },
			ConstructNNParam{ { 6, 9 }, { Ins(7, 1), Ins(4, 1), Ins(6, 1) } }
		}
	},

	{
		ParamType::INS_AT_CORNER_CASES,
		{
			ConstructNNParam{ { 0, 0 }, { Ins(7, 0) } },
			ConstructNNParam{ { 0, 3 }, { Ins(1, 1) } },
			ConstructNNParam{ { 2, 0 }, { Ins(2, 0) } },
			ConstructNNParam{ { 2, 3 }, { Ins(0, 1) } },
			ConstructNNParam{ { 0, 0 }, { Ins(0, 1) } },
			ConstructNNParam{ { 0, 0 }, { Ins(2, 0), Ins(0, 2), Ins(1, 1), Ins(0, 0) } },
			ConstructNNParam{ { 0, 0 }, { Ins(0, 0), Ins(0, 2), Ins(0, 1), Ins(0, 0) } },
			ConstructNNParam{ { 1, 1 }, { Ins(1, 1), Ins(1, 2), Ins(1, 3), Ins(1, 4), Ins(1, 5) } },
			ConstructNNParam{ { 1, 1 }, { Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0), Ins(1, 0) } }
		}
	},

	{
		ParamType::INS_AT_MULT,
		{
			ConstructNNParam{ { 7, 0, 8 }, { Ins(4, 0) } },
			ConstructNNParam{ { 6, 0, 0, 9 }, { Ins(7, 1) } },
			ConstructNNParam{ { 6, 1, 9, 0, 9 }, { Ins(7, 1), Ins(4, 1) } },
			ConstructNNParam{ { 6, 3, 3, 3, 9 }, { Ins(7, 1), Ins(4, 2) } },
			ConstructNNParam{ { 6, 0, 5, 0, 9 }, { Ins(7, 1), Ins(4, 1), Ins(6, 1) } }
		}
	},
	
	{
		ParamType::INS_DIFF_ACTIVATIONS,
		{
			ConstructNNParam{ { 7, 0 }, { Ins(4, ActivationFType::ReLU) } },
			ConstructNNParam{ { 6, 0 }, { Ins(7, 1, ActivationFType::ReLU) } },
			ConstructNNParam{ { 6, 1 }, { Ins(7, 1, ActivationFType::tanh), Ins(4, 1) } },
			ConstructNNParam{ { 6, 0 }, { Ins(7, 1), Ins(4, 1, ActivationFType::softplus), Ins(6, 1, ActivationFType::ReLU) } }
		}
	},	

	{
		ParamType::STRESS_TESTS,
		{
			ConstructNNParam{ { 10000, 10000 }, { Ins(7) } },
			ConstructNNParam{ { 2, 2 }, std::vector<Ins>(20, Ins(1000)) },
			ConstructNNParam{ { 1000, 1000 }, std::vector<Ins>(20, Ins(1000)) },
			ConstructNNParam{ { 1000, 1000 }, std::vector<Ins>(20, Ins(1000, 0)) },
			ConstructNNParam{ std::vector<LayerMetadata>(25, 1000), {} },
		}
	}
};
