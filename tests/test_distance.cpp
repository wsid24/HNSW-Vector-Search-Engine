#include "hnsw/distance.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace hnsw;

void assert_close(float a, float b, float tol = 1e-5f) {
    if (std::abs(a - b) > tol) {
        std::cerr << "Assertion failed: " << a << " != " << b << std::endl;
        assert(false);
    }
}

int main() {
    std::cout << "Running distance tests..." << std::endl;

    // Test 1: Identical vectors
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {1.0f, 2.0f, 3.0f};
        
        assert_close(l2_squared_distance(a.data(), b.data(), a.size()), 0.0f);
        assert_close(cosine_distance(a.data(), b.data(), a.size()), 0.0f);
    }

    // Test 2: Orthogonal vectors
    {
        std::vector<float> a = {1.0f, 0.0f, 0.0f};
        std::vector<float> b = {0.0f, 1.0f, 0.0f};
        
        assert_close(l2_squared_distance(a.data(), b.data(), a.size()), 2.0f);
        assert_close(dot_product_distance(a.data(), b.data(), a.size()), 0.0f);
        assert_close(cosine_distance(a.data(), b.data(), a.size()), 1.0f);
    }
    
    // Test 3: Hand-computed 3D example
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {4.0f, 5.0f, 6.0f};
        
        // L2 squared = (1-4)^2 + (2-5)^2 + (3-6)^2 = (-3)^2 + (-3)^2 + (-3)^2 = 9 + 9 + 9 = 27
        assert_close(l2_squared_distance(a.data(), b.data(), a.size()), 27.0f);
        
        // Dot product distance = -(1*4 + 2*5 + 3*6) = -(4 + 10 + 18) = -32
        assert_close(dot_product_distance(a.data(), b.data(), a.size()), -32.0f);
        
        // Norm A = sqrt(1+4+9) = sqrt(14)
        // Norm B = sqrt(16+25+36) = sqrt(77)
        // Cosine similarity = 32 / (sqrt(14) * sqrt(77))
        float sim = 32.0f / (std::sqrt(14.0f) * std::sqrt(77.0f));
        assert_close(cosine_distance(a.data(), b.data(), a.size()), 1.0f - sim);
    }

    // Test 4: Relationship between cosine distance and dot product on normalized vectors
    {
        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {4.0f, -5.0f, 6.0f};
        
        // Normalize a
        float norm_a = 0.0f;
        for (float v : a) norm_a += v * v;
        norm_a = std::sqrt(norm_a);
        for (float& v : a) v /= norm_a;
        
        // Normalize b
        float norm_b = 0.0f;
        for (float v : b) norm_b += v * v;
        norm_b = std::sqrt(norm_b);
        for (float& v : b) v /= norm_b;
        
        float neg_dot = dot_product_distance(a.data(), b.data(), a.size());
        float cos_dist = cosine_distance(a.data(), b.data(), a.size());
        
        // For normalized vectors, cosine_similarity = dot_product
        // And cosine_distance = 1 - cosine_similarity = 1 - dot_product = 1 + (-dot_product)
        assert_close(cos_dist, 1.0f + neg_dot);
    }

    // Test 5: Sign convention for dot_product_distance
    {
        std::vector<float> a = {1.0f, 0.0f, 0.0f};
        std::vector<float> b_similar = {0.9f, 0.1f, 0.0f};
        std::vector<float> b_opposite = {-0.9f, 0.1f, 0.0f};

        float dist_similar = dot_product_distance(a.data(), b_similar.data(), a.size());
        float dist_opposite = dot_product_distance(a.data(), b_opposite.data(), a.size());

        // Closer vectors should have a smaller distance value
        if (dist_similar >= dist_opposite) {
            std::cerr << "Assertion failed: similar distance " << dist_similar 
                      << " is not less than opposite distance " << dist_opposite << std::endl;
            assert(false);
        }
    }

    std::cout << "All distance tests passed!" << std::endl;
    return 0;
}
