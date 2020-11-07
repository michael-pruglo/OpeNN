#include <OpeNN/package/io/nn_printer.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace openn;

namespace
{
	class NeuralNetworkPrinter
	{
	public:
		explicit			NeuralNetworkPrinter(const NeuralNetwork& nn);
				std::string getDebugString() const;
	private:
				std::string getHeader() const;
				std::string getBody() const;
				std::string getFooter() const;
		static	std::string getLine(size_t len);
				size_t		getHeight() const;

	private:
		const NeuralNetwork& nn;
		const size_t HGAP = 5, LAYER_W = 9, TOTAL_W;
	};
}

std::ostream& openn::operator<<(std::ostream& os, const NeuralNetwork& nn)
{
	return os << NeuralNetworkPrinter(nn).getDebugString();
}



namespace
{
	NeuralNetworkPrinter::NeuralNetworkPrinter(const NeuralNetwork& nn_) 
		: nn(nn_)
		, TOTAL_W((HGAP + LAYER_W) * nn.layers.size())
	{}

	std::string NeuralNetworkPrinter::getDebugString() const
	{
		return 
			getHeader() + "\n" +
			getBody() + "\n" +
			getFooter();
	}

	std::string NeuralNetworkPrinter::getHeader() const { return getLine(TOTAL_W + HGAP); }
	std::string NeuralNetworkPrinter::getFooter() const { return getHeader(); }
	std::string NeuralNetworkPrinter::getLine(size_t len) { return std::string(len, '='); }

	std::string NeuralNetworkPrinter::getBody() const
	{
		std::ostringstream ss;
		for (size_t i = 0; i < getHeight(); ++i)
		{
			for (const auto& layer: nn.layers)
			{
				ss << std::string(HGAP, ' ');
				if (i < layer.size())
					ss << "(" << std::setw(2) << layer[i].w.size() << ") " << std::fixed << std::setprecision(2) << layer[i].bias;
				else 
					ss << std::string(LAYER_W, ' ');
			}
			ss << "\n";
		}
		return ss.str();
	}
	
	size_t NeuralNetworkPrinter::getHeight() const
	{
		const auto& it_to_longest = std::max_element(nn.layers.begin(), nn.layers.end(), 
			[](const Layer& l1, const Layer& l2){ return l1.size() < l2.size(); }
		);
		return it_to_longest->size();
	}
}