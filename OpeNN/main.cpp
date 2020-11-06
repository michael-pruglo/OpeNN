#include <iomanip>
#include <iostream>
#include "package/opeNN.hpp"
#include "package/io/nn_serializer.hpp"

int main()
{
	openn::NeuralNetwork nn(3, 9);
	nn.addLayer(7);
	nn.addLayer(6);
	std::cout << std::setw(4) << to_json(nn);
}
