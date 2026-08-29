#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

using namespace hnsw;

int main() {
    std::cout << "Running persistence test..." << std::endl;

    const size_t dim = 32;
    const size_t num_vectors = 500;
    const size_t num_queries = 10;
    const size_t k = 10;
    const size_t efSearch = 50;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    HNSWIndex index(dim, 16, 200, 42);
    
    std::vector<float> vec(dim);
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            vec[d] = dist(rng);
        }
        index.insert(vec.data());
    }
    
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            queries[i][d] = dist(rng);
        }
    }
    
    std::vector<std::vector<std::pair<float, uint32_t>>> original_results(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        original_results[q] = index.search(queries[q].data(), k, efSearch);
    }
    
    std::string temp_file = "test_index_persistence.bin";
    std::cout << "Saving index to " << temp_file << "..." << std::endl;
    index.save(temp_file);
    
    std::cout << "Loading index into a new instance..." << std::endl;
    // We can instantiate with dummy params, they will be overwritten by load
    HNSWIndex loaded_index(dim); 
    loaded_index.load(temp_file);
    
    // Assert structural equality
    if (loaded_index.nodes_.size() != index.nodes_.size()) {
        throw std::runtime_error("Node count mismatch");
    }
    if (loaded_index.entry_point_id_ != index.entry_point_id_) {
        throw std::runtime_error("Entry point mismatch");
    }
    if (loaded_index.max_layer_ != index.max_layer_) {
        throw std::runtime_error("Max layer mismatch");
    }
    if (loaded_index.vector_store_.size() != index.vector_store_.size()) {
        throw std::runtime_error("Vector store size mismatch");
    }
    
    std::cout << "Structural assertions passed!" << std::endl;
    
    // Test search on loaded index
    for (size_t q = 0; q < num_queries; ++q) {
        auto loaded_results = loaded_index.search(queries[q].data(), k, efSearch);
        
        if (loaded_results.size() != original_results[q].size()) {
            throw std::runtime_error("Search result size mismatch");
        }
        
        for (size_t i = 0; i < loaded_results.size(); ++i) {
            if (loaded_results[i].first != original_results[q][i].first ||
                loaded_results[i].second != original_results[q][i].second) {
                throw std::runtime_error("Search result mismatch at rank " + std::to_string(i));
            }
        }
    }
    
    std::cout << "Search results are bit-for-bit identical!" << std::endl;
    
    // Clean up
    std::remove(temp_file.c_str());
    
    // Test negative cases for load()
    std::cout << "Testing load() failure paths..." << std::endl;
    
    bool caught_missing = false;
    try {
        HNSWIndex dummy_index(dim);
        dummy_index.load("does_not_exist_at_all.bin");
    } catch (const std::runtime_error& e) {
        caught_missing = true;
    }
    if (!caught_missing) {
        throw std::runtime_error("Failed to throw on missing file");
    }
    std::cout << "Successfully caught load() of missing file." << std::endl;
    
    std::string garbage_file = "garbage_test.bin";
    {
        std::ofstream out(garbage_file, std::ios::binary);
        out.write("1234567890", 10);
    }
    
    bool caught_garbage = false;
    try {
        HNSWIndex dummy_index(dim);
        dummy_index.load(garbage_file);
    } catch (const std::runtime_error& e) {
        caught_garbage = true;
    }
    std::remove(garbage_file.c_str());
    
    if (!caught_garbage) {
        throw std::runtime_error("Failed to throw on garbage file");
    }
    std::cout << "Successfully caught load() of garbage file." << std::endl;
    
    std::cout << "Persistence test passed successfully." << std::endl;
    return 0;
}
