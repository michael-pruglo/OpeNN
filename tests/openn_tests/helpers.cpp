#include <tests/openn_tests/helpers.hpp>
#include <core/random.hpp>

namespace openn
{
    ActivationFType rand_activation()
    {
        const auto sz = static_cast<size_t>(ActivationFType::_SIZE);
        const auto idx = core::rand_i<size_t>(0, sz-1U);
        return static_cast<const ActivationFType>(idx);
    }
}
