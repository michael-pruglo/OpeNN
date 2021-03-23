#pragma once

#include <openn/types.hpp>

namespace openn
{
    class NeuralNetworkPrinter
    {
    public:
        explicit NeuralNetworkPrinter(const FeedForwardNetwork& nn);
        std::string getDebugString() const;
    private:
        std::string getHeader() const;
        std::string getBody() const;
        std::string getFooter() const;
        static std::string getLine(size_t len);
        std::string getActivationLine() const;
        size_t		getHeight() const;

    private:
        const FeedForwardNetwork& nn;
        const size_t HGAP = 5, LAYER_W = 11, TOTAL_W;
    };

    std::ostream& operator<<(std::ostream & os, const FeedForwardNetwork& nn);

    std::string to_string(const ActivationFType& activation_type);
    ActivationFType string_to_activation_type(const std::string& s);
}

