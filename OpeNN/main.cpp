#include <iomanip>
#include <iostream>
#include "package/opeNN.hpp"
#include "package/io/nn_printer.hpp"
#include "package/io/nn_serializer.hpp"

int main()
{
	openn::NeuralNetwork nn(3, 9);
	nn.addLayer(7);
	nn.addLayer(6);
	const auto& nn_json = nlohmann::json(nn);
	std::cout<< nn <<"\n";
	std::cout << std::setw(4) << nn_json;

	const openn::NeuralNetwork nn2(nn_json);
	std::cout << nn2;
}
