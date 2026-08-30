#include "hnsw/distance.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

using namespace hnsw;

void assert_close(float a, float b, float tol = 1e-3f) {
    if (std::abs(a - b) > tol * std::max(1.0f, std::max(std::abs(a), std::abs(b)))) {
        std::cerr << "Assertion failed: " << a << " != " << b << std::endl;
        assert(false);
    }
}

int main() {
    std::cout << "Running NEON distance test..." << std::endl;

#if defined(__ARM_NEON) || defined(__aarch64__)
    std::cout << "NEON is available. Testing l2_squared_distance_neon..." << std::endl;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    // Test multiple of 4 (dim = 384)
    {
        size_t dim = 384;
        for (int iter = 0; iter < 1000; ++iter) {
            std::vector<float> a(dim), b(dim);
            for (size_t d = 0; d < dim; ++d) {
                a[d] = dist(rng);
                b[d] = dist(rng);
            }
            float scalar_dist = l2_squared_distance(a.data(), b.data(), dim);
            float neon_dist = l2_squared_distance_neon(a.data(), b.data(), dim);
            assert_close(scalar_dist, neon_dist);
        }
        std::cout << "Verified 1000 random pairs at dim=384." << std::endl;
    }
    
    // Test non-multiple of 4 (dim = 13)
    {
        size_t dim = 13;
        for (int iter = 0; iter < 100; ++iter) {
            std::vector<float> a(dim), b(dim);
            for (size_t d = 0; d < dim; ++d) {
                a[d] = dist(rng);
                b[d] = dist(rng);
            }
            float scalar_dist = l2_squared_distance(a.data(), b.data(), dim);
            float neon_dist = l2_squared_distance_neon(a.data(), b.data(), dim);
            assert_close(scalar_dist, neon_dist);
        }
        std::cout << "Verified 100 random pairs at dim=13 (scalar remainder path)." << std::endl;
    }
#else
    std::cout << "NEON is not available. Skipping NEON-specific tests." << std::endl;
#endif

    std::cout << "All distance tests passed!" << std::endl;
    return 0;
}
