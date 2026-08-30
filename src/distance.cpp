#include "hnsw/distance.hpp"
#include <cmath>
#include <stdexcept>

namespace hnsw {

float l2_squared_distance(const float* a, const float* b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

float l2_squared_distance_neon(const float* a, const float* b, size_t dim) {
    size_t i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    // Process 4 floats at a time
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_vec = vfmaq_f32(sum_vec, diff, diff);
    }
    
    // Horizontally sum the vector
    float dist = vaddvq_f32(sum_vec);
    
    // Remainder loop
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    
    return dist;
}
#endif

float l2_squared_distance_best(const float* a, const float* b, size_t dim) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    return l2_squared_distance_neon(a, b, dim);
#else
    return l2_squared_distance(a, b, dim);
#endif
}

float dot_product_distance(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return -dot;
}

float cosine_distance(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    
    if (norm_a_sq == 0.0f || norm_b_sq == 0.0f) {
        // Handle zero vectors if necessary, typically returning a default value
        return 1.0f; 
    }
    
    float similarity = dot / (std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq));
    
    // Clamp to [-1, 1] to avoid float precision issues
    if (similarity > 1.0f) similarity = 1.0f;
    if (similarity < -1.0f) similarity = -1.0f;
    
    return 1.0f - similarity; // 0 means identical direction, 2 means opposite
}

} // namespace hnsw
