#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <cassert>

using namespace hnsw;

int main() {
    std::cout << "Running search correctness test..." << std::endl;

    const size_t dim = 128;
    const size_t num_vectors = 1000;
    const size_t num_queries = 10;
    const size_t k = 10;
    const size_t efSearch = 200;

    // Parse ground truth file
    std::string gt_path = std::string(PROJECT_ROOT_DIR) + "/data/ground_truth.txt";
    std::ifstream in_file(gt_path);
    if (!in_file) {
        std::cerr << "Failed to open " << gt_path << ". Did you run main to generate it?" << std::endl;
        return 1;
    }

    std::vector<std::unordered_set<uint32_t>> ground_truth(num_queries);
    std::string line;
    int q_count = 0;
    while (std::getline(in_file, line) && q_count < num_queries) {
        // Format: "query_id: id1:dist1, id2:dist2, ..."
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string ids_part = line.substr(colon_pos + 1);
        std::stringstream ss(ids_part);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            // Trim spaces
            size_t start = token.find_first_not_of(" \t");
            if (start != std::string::npos) token = token.substr(start);
            
            size_t colon = token.find(':');
            if (colon != std::string::npos) {
                uint32_t id = std::stoul(token.substr(0, colon));
                ground_truth[q_count].insert(id);
            }
        }
        q_count++;
    }
    in_file.close();
    
    if (q_count != num_queries) {
        std::cerr << "Failed to read all queries from ground truth!" << std::endl;
        return 1;
    }

    // Generate exactly the same dataset and queries as main.cpp
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    HNSWIndex index(dim, 16, 200, 42);

    std::cout << "Building HNSW index with " << num_vectors << " vectors..." << std::endl;
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

    std::cout << "Running HNSW search for " << num_queries << " queries (ef=" << efSearch << ")..." << std::endl;
    double total_recall = 0.0;
    
    for (size_t q = 0; q < num_queries; ++q) {
        auto results = index.search(queries[q].data(), k, efSearch);
        
        int matches = 0;
        for (const auto& r : results) {
            if (ground_truth[q].find(r.second) != ground_truth[q].end()) {
                matches++;
            }
        }
        
        double recall = static_cast<double>(matches) / k;
        total_recall += recall;
        
        std::cout << "Query " << q << " Recall@" << k << ": " << recall 
                  << " (" << matches << "/" << k << ")" << std::endl;
    }
    
    double avg_recall = total_recall / num_queries;
    std::cout << "Average Recall@" << k << ": " << avg_recall << std::endl;
    
    // Assert high recall (at least 95%)
    if (avg_recall < 0.95) {
        std::cerr << "Assertion failed: Average recall " << avg_recall << " is less than 0.95" << std::endl;
        assert(false);
    }

    std::cout << "Search correctness verified successfully!" << std::endl;
    return 0;
}
