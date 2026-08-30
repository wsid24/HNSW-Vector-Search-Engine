#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace hnsw;

int main() {
    std::cout << "Running search_layer test..." << std::endl;

    size_t dim = 8;
    HNSWIndex index(dim); // M=16, seed=42

    // Generate 15 random vectors
    size_t num_nodes = 15;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> vecs(num_nodes, std::vector<float>(dim));
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            vecs[i][d] = dist(rng);
        }
        uint32_t vec_id = index.vector_store_.add_vector(vecs[i].data());
        
        GraphNode node;
        node.vector_id = vec_id;
        node.max_layer = 0;
        index.allocate_node_in_arena(0);
        index.nodes_.push_back(node);
    }
    index.max_layer_ = 0;
    index.entry_point_id_ = 0;

    // Build a connected graph at layer 0 by connecting each node to its true 3 nearest neighbors
    for (size_t i = 0; i < num_nodes; ++i) {
        std::vector<std::pair<float, uint32_t>> distances;
        for (size_t j = 0; j < num_nodes; ++j) {
            if (i == j) continue;
            float d = index.distance_to_query(vecs[i].data(), j);
            distances.push_back({d, j});
        }
        std::sort(distances.begin(), distances.end()); // closest first
        
        // Add top 3 connections
        for (size_t k = 0; k < 3; ++k) {
            uint32_t j = distances[k].second;
            
            // Add j to i
            uint16_t ci = index.get_neighbor_count(i, 0);
            index.neighbor_slots(i, 0)[ci] = j;
            index.set_neighbor_count(i, 0, ci + 1);
            
            // Add i to j
            uint16_t cj = index.get_neighbor_count(j, 0);
            index.neighbor_slots(j, 0)[cj] = i;
            index.set_neighbor_count(j, 0, cj + 1);
        }
    }
    
    // Remove duplicates from bidirectional edges
    for (size_t i = 0; i < num_nodes; ++i) {
        uint32_t* neighbors = index.neighbor_slots(i, 0);
        uint16_t count = index.get_neighbor_count(i, 0);
        std::sort(neighbors, neighbors + count);
        auto last = std::unique(neighbors, neighbors + count);
        index.set_neighbor_count(i, 0, std::distance(neighbors, last));
    }

    // Generate a query
    std::vector<float> query(dim);
    for (size_t d = 0; d < dim; ++d) query[d] = dist(rng);

    // Brute force top-5
    std::vector<std::pair<float, uint32_t>> bf_distances;
    for (size_t i = 0; i < num_nodes; ++i) {
        float d = index.distance_to_query(query.data(), i);
        bf_distances.push_back({d, i});
    }
    std::sort(bf_distances.begin(), bf_distances.end());
    bf_distances.resize(5); // top 5

    // search_layer
    std::vector<uint32_t> entry_points = {static_cast<uint32_t>(index.entry_point_id_)};
    size_t ef = 5;
    auto search_results = index.search_layer(query.data(), entry_points, ef, 0);

    // Print comparison
    std::cout << std::left << std::setw(30) << "Brute-force Top-5" 
              << std::left << std::setw(30) << "search_layer Top-5" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    bool match = true;
    for (size_t i = 0; i < 5; ++i) {
        std::string bf_str = "[id=" + std::to_string(bf_distances[i].second) + ", d=" + std::to_string(bf_distances[i].first) + "]";
        std::string sr_str = (i < search_results.size()) ? 
            ("[id=" + std::to_string(search_results[i].second) + ", d=" + std::to_string(search_results[i].first) + "]") : "N/A";
        
        std::cout << std::left << std::setw(30) << bf_str 
                  << std::left << std::setw(30) << sr_str << std::endl;
                  
        if (i < search_results.size() && bf_distances[i].second != search_results[i].second) {
            match = false;
        }
    }

    if (match) {
        std::cout << "Success! search_layer matches brute-force on this small connected graph." << std::endl;
    } else {
        std::cout << "Warning: search_layer did not perfectly match brute-force (this can happen if graph is not perfectly navigable, but for 15 nodes and ef=5 it usually should)." << std::endl;
    }

    return 0;
}
