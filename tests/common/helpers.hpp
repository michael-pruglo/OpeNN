#pragma once

#include <gtest/gtest.h>

#define EXPECT_CONTAINER_EQ_IMPL(GOOGLETEST_LINE) \
{  \
    ASSERT_EQ(a1.size(), a2.size());  \
    auto it1 = a1.begin();  \
    auto it2 = a2.begin();  \
    for ( ; it1 != a1.end(); ++it1, ++it2)  \
    {  \
        GOOGLETEST_LINE  \
            << "i = " << std::distance(a1.begin(), it1) << "\n"  \
            ;  \
    }  \
}



template<typename Iterable1, typename Iterable2 = Iterable1>
inline void expect_container_eq(const Iterable1& a1, const Iterable2& a2)
{
    EXPECT_CONTAINER_EQ_IMPL(
        EXPECT_DOUBLE_EQ(*it1, *it2)
    )
}

template<typename Iterable1, typename Iterable2 = Iterable1>
inline void expect_container_eq(const Iterable1& a1, const Iterable2& a2, double tolerance)
{
    EXPECT_CONTAINER_EQ_IMPL(
        EXPECT_NEAR(*it1, *it2, tolerance)
        )
}
