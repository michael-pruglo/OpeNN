#pragma once

#include <OpeNN/package/types.hpp>

namespace openn
{
	std::ostream& operator<<(std::ostream & os, const NeuralNetwork& nn);
	
	std::string to_string(const ActivationFType& activation_type);
	ActivationFType string_to_activation_type(const std::string& s);
}

