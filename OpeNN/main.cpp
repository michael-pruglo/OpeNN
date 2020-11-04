#include <iostream>
#include "opeNN.hpp"

int main()
{
	openn::NeuralNetwork nn(0, 9);
	nn.addLayer(7);
	std::cout << nn;
}
