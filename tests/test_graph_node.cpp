#include "hnsw/graph_node.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace hnsw;

void assert_close(float a, float b, float tol = 1e-5f) {
    if (std::abs(a - b) > tol) {
        std::cerr << "Assertion failed: " << a << " != " << b << std::endl;
        assert(false);
    }
}

int main() {
    std::cout << "Running graph node and vector store tests..." << std::endl;

    size_t dim = 3;
    VectorStore store(dim);
    
    assert(store.size() == 0);
    assert(store.dim() == 3);

    // Add 5 vectors
    std::vector<std::vector<float>> test_vectors = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f},
        {-1.0f, -2.0f, -3.0f},
        {0.0f, 0.0f, 0.0f}
    };

    for (size_t i = 0; i < test_vectors.size(); ++i) {
        uint32_t id = store.add_vector(test_vectors[i].data());
        assert(id == i);
    }

    assert(store.size() == 5);

    // Verify vectors
    for (size_t i = 0; i < test_vectors.size(); ++i) {
        const float* retrieved = store.get_vector(i);
        for (size_t d = 0; d < dim; ++d) {
            assert_close(retrieved[d], test_vectors[i][d]);
        }
    }
    
    // Create some GraphNode instances manually
    GraphNode node0;
    node0.vector_id = 0;
    node0.max_layer = 1;
    
    GraphNode node1;
    node1.vector_id = 1;
    node1.max_layer = 0;
    
    assert(node0.vector_id == 0);
    assert(node0.max_layer == 1);

    assert(node1.vector_id == 1);
    assert(node1.max_layer == 0);

    // Reallocation safety test
    {
        VectorStore reserved_store(3, 1000);
        std::vector<float> first_vec = {1.1f, 2.2f, 3.3f};
        reserved_store.add_vector(first_vec.data());
        
        // Capture pointer to first vector
        const float* first_ptr = reserved_store.get_vector(0);
        
        // Add many vectors (e.g. 500) to approach but not exceed reserved capacity
        for (int i = 0; i < 500; ++i) {
            std::vector<float> dummy = {static_cast<float>(i), 0.0f, 0.0f};
            reserved_store.add_vector(dummy.data());
        }
        
        // Confirm originally captured pointer still reads correctly
        assert_close(first_ptr[0], 1.1f);
        assert_close(first_ptr[1], 2.2f);
        assert_close(first_ptr[2], 3.3f);
    }
    
    // Out-of-range test
    {
        VectorStore small_store(3);
        std::vector<float> vec = {1.0f, 1.0f, 1.0f};
        small_store.add_vector(vec.data());
        
        bool threw = false;
        try {
            // size is 1, so getting id 1 or greater should throw
            small_store.get_vector(1);
        } catch (const std::out_of_range& e) {
            threw = true;
            std::cout << "Caught expected exception: " << e.what() << std::endl;
        }
        assert(threw);
    }

    std::cout << "All graph node tests passed!" << std::endl;
    return 0;
}
