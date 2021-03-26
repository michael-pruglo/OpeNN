#pragma once

#include <gtest/gtest.h>

inline void expect_double_vec_eq(const std::vector<double>& v1, const std::vector<double>& v2)
{
    ASSERT_EQ(v1.size(), v2.size());
    for (int i = 0; i < v1.size(); ++i)
        EXPECT_DOUBLE_EQ(v1[i], v2[i]) << "i = " << i;
}

inline void expect_double_vec_eq(const std::vector<double>& v1, const std::vector<double>& v2, double tolerance)
{
    ASSERT_EQ(v1.size(), v2.size());
    for (int i = 0; i < v1.size(); ++i)
        EXPECT_NEAR(v1[i], v2[i], tolerance) << "i = " << i;
}

inline void expect_double_matrix_eq(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2)
{
    ASSERT_EQ(m1.size(), m2.size());
    for (size_t i = 0; i < m1.size(); ++i)
        expect_double_vec_eq(m1[i], m2[i]);
}

inline void expect_double_matrix_eq(const std::vector<std::vector<double>>& m1, const std::vector<std::vector<double>>& m2, double tolerance)
{
    ASSERT_EQ(m1.size(), m2.size());
    for (size_t i = 0; i < m1.size(); ++i)
        expect_double_vec_eq(m1[i], m2[i], tolerance);
}