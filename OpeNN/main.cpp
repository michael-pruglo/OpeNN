#include <iostream>
#include "opeNN.hpp"

int main()
{
	openn::NeuralNetwork nn(2, 2);
	nn.addLayer(3);
	std::cout << nn;
}
