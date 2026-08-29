#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <map>
#include <cassert>
#include <cmath>

using namespace hnsw;

int main() {
    std::cout << "Running layer assignment test..." << std::endl;

    size_t dim = 128;
    size_t M = 16;
    HNSWIndex index(dim, M, 42); // fixed seed

    int num_samples = 10000;
    std::map<int, int> histogram;

    for (int i = 0; i < num_samples; ++i) {
        int layer = index.assign_random_layer();
        histogram[layer]++;
    }

    std::cout << "Histogram over " << num_samples << " calls:" << std::endl;
    for (const auto& pair : histogram) {
        std::cout << "Layer " << pair.first << ": " << pair.second << std::endl;
    }

    // Check assertions
    // Expected probability of layer 0 is 1 - 1/M = 15/16
    // So for 10000 samples, we expect around 9375 at layer 0.
    double expected_layer_0 = num_samples * (1.0 - 1.0 / M); // 9375
    int actual_layer_0 = histogram[0];
    
    // Nodes at layer >= 1
    int actual_layer_ge_1 = num_samples - actual_layer_0;
    double expected_layer_ge_1 = num_samples * (1.0 / M); // 625

    std::cout << "Expected layer 0: ~" << expected_layer_0 << ", Actual: " << actual_layer_0 << std::endl;
    std::cout << "Expected layer >= 1: ~" << expected_layer_ge_1 << ", Actual: " << actual_layer_ge_1 << std::endl;

    // 20% tolerance
    double tol_0 = expected_layer_0 * 0.20;
    double tol_ge_1 = expected_layer_ge_1 * 0.20;

    if (std::abs(actual_layer_0 - expected_layer_0) > tol_0) {
        std::cerr << "Assertion failed: Layer 0 count " << actual_layer_0 
                  << " is not within 20% of " << expected_layer_0 << std::endl;
        assert(false);
    }
    
    if (std::abs(actual_layer_ge_1 - expected_layer_ge_1) > tol_ge_1) {
        std::cerr << "Assertion failed: Layer >= 1 count " << actual_layer_ge_1 
                  << " is not within 20% of " << expected_layer_ge_1 << std::endl;
        assert(false);
    }

    std::cout << "All layer assignment tests passed!" << std::endl;
    return 0;
}
