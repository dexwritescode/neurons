#include "simd_utils.h"
#include <arm_neon.h>

/*
  Key NEON intrinsics used:
  - vld1q_f32(): Load 4 floats into a vector
  - vfmaq_f32(): Fused multiply-add (sum += a * b)
  - vaddvq_f32(): Sum all elements in the vector
  - vdupq_n_f32(): Create vector with all elements set to same value
*/

float simd_dot_product(const float* a, const float* b, size_t size) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time
    size_t simd_size = size & ~3; // Round down to multiple of 4
    
    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t a_vec = vld1q_f32(&a[i]);
        float32x4_t b_vec = vld1q_f32(&b[i]);
        sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec); // Fused multiply-add
    }
    
    // Sum the vector elements
    float result = vaddvq_f32(sum_vec);
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}
