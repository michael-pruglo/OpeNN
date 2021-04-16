#pragma once

#include<src/openn/training.hpp>

namespace projects::simplest
{
    class XorEnvironment
    {
    public:
        void run();

    private:
        openn::FeedForwardNetworkTrainer trainer;
    };
}

