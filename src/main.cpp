#include <OpeNN/OpeNN/opeNN.hpp>
#include <OpeNN/OpeNN/io/nn_printer.hpp>
#include <OpeNN/OpeNN/io/nn_serializer.hpp>
#include <iostream>
#include <iomanip>

int main()
{
    openn::NeuralNetwork nn({3, 7, 6, 9});
    std::cout<< nn <<"\n";
    std::cout << std::setw(4) << nlohmann::json(nn);
}
