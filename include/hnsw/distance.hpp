#pragma once

#include <cstddef>

namespace hnsw {

// Computes the L2 (Euclidean) squared distance between two float vectors.
float l2_squared_distance(const float* a, const float* b, size_t dim);

#if defined(__ARM_NEON) || defined(__aarch64__)
// NEON-accelerated L2 squared distance
float l2_squared_distance_neon(const float* a, const float* b, size_t dim);
#endif

// Dispatcher function that uses the best available implementation
float l2_squared_distance_best(const float* a, const float* b, size_t dim);

// Computes the dot product distance between two float vectors.
// Returns the negative dot product so that smaller values represent more similar (closer) vectors,
// matching the contract of the other distance functions.
float dot_product_distance(const float* a, const float* b, size_t dim);

// Computes the cosine distance (1 - cosine_similarity) between two float vectors.
// Cosine distance is in [0, 2], where 0 is identical direction, 2 is opposite direction.
float cosine_distance(const float* a, const float* b, size_t dim);

} // namespace hnsw
