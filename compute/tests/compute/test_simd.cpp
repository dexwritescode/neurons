#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "compute/backends/simd/simd_utils.h"

class SimdTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup common test data
        a = {1.0f, 2.0f, 3.0f, 4.0f};
        b = {2.0f, 3.0f, 4.0f, 5.0f};
    }
    
    std::vector<float> a, b;
};

TEST_F(SimdTest, DotProductBasicTest) {
    float result = simd_dot_product(a.data(), b.data(), 4);
    EXPECT_FLOAT_EQ(result, 40.0f);
}
