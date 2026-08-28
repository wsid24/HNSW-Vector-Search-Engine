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
