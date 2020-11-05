#include <iostream>
#include "opeNN.hpp"

int main()
{
	openn::NeuralNetwork nn(3, 9);
	nn.addLayer(7);
	nn.addLayer(6);
	std::cout << nn;
}
