#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <cassert>

using namespace hnsw;

int main() {
    std::cout << "Running insert test..." << std::endl;

    size_t dim = 8;
    size_t M = 16;
    size_t efConstruction = 200;
    
    // Construct index with reserved capacity inside VectorStore indirectly 
    // Wait, HNSWIndex constructor doesn't reserve yet. The user prompt says "no reservation yet — that comes later when insert is built". 
    // Wait, HNSWIndex constructor currently calls `vector_store_(dim)`. So no reservation is done. This is correct as per instructions.
    
    HNSWIndex index(dim, M, efConstruction, 42); 

    size_t num_nodes = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vec(dim);
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = dist(rng);
        }
        index.insert(vec.data());
    }

    // Verify invariants
    std::map<int, int> histogram;
    
    for (size_t i = 0; i < index.nodes_.size(); ++i) {
        const auto& node = index.nodes_[i];
        
        // 1. Every node has neighbors.size() == max_layer + 1
        assert(node.neighbors.size() == node.max_layer + 1);
        
        // Populate histogram
        histogram[node.max_layer]++;
        
        // 2. Constraints on neighbor list size
        for (int layer = 0; layer <= node.max_layer; ++layer) {
            size_t M_max = (layer == 0) ? (2 * M) : M;
            if (node.neighbors[layer].size() > M_max) {
                std::cerr << "Assertion failed: Node " << i << " at layer " << layer 
                          << " has " << node.neighbors[layer].size() << " neighbors, expected <= " << M_max << std::endl;
                assert(false);
            }
        }
    }
    
    // 3. Layer 0 must have all 1000 nodes conceptually.
    // In our implementation, since every node has a max_layer >= 0, it is guaranteed
    // to have a neighbors[0] vector and exist at layer 0. 
    assert(index.nodes_.size() == num_nodes);

    std::cout << "Insert structural invariants verified successfully!" << std::endl;
    
    std::cout << "Histogram of max_layer across " << num_nodes << " nodes:" << std::endl;
    for (const auto& pair : histogram) {
        std::cout << "Layer " << pair.first << " (max_layer): " << pair.second << " nodes" << std::endl;
    }
    
    std::cout << "All insert tests passed!" << std::endl;
    return 0;
}
