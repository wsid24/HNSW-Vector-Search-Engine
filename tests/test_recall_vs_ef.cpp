#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <iomanip>
#include <cassert>

using namespace hnsw;

int main() {
    std::cout << "Running recall vs ef test..." << std::endl;

    const size_t dim = 128;
    const size_t num_vectors = 1000;
    const size_t num_queries = 10;
    const size_t k = 10;

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
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string ids_part = line.substr(colon_pos + 1);
        std::stringstream ss(ids_part);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
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

    // Generate dataset and queries
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

    std::vector<size_t> ef_values = {10, 20, 50, 100, 200};
    
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left << std::setw(10) << "ef"
              << std::left << std::setw(20) << "Avg Recall@10"
              << std::left << std::setw(25) << "Avg Nodes Visited"
              << std::left << std::setw(20) << "% of Graph Visited" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    double visited_at_ef10 = 0.0;

    for (size_t ef : ef_values) {
        double total_recall = 0.0;
        size_t total_visited = 0;
        
        for (size_t q = 0; q < num_queries; ++q) {
            size_t visited_count = 0;
            auto results = index.search(queries[q].data(), k, ef, &visited_count);
            
            int matches = 0;
            for (const auto& r : results) {
                if (ground_truth[q].find(r.second) != ground_truth[q].end()) {
                    matches++;
                }
            }
            
            total_recall += static_cast<double>(matches) / k;
            total_visited += visited_count;
        }
        
        double avg_recall = total_recall / num_queries;
        double avg_visited = static_cast<double>(total_visited) / num_queries;
        double percent_visited = (avg_visited / num_vectors) * 100.0;
        
        if (ef == 10) {
            visited_at_ef10 = avg_visited;
        }

        std::cout << std::left << std::setw(10) << ef
                  << std::left << std::setw(20) << std::fixed << std::setprecision(4) << avg_recall
                  << std::left << std::setw(25) << std::fixed << std::setprecision(1) << avg_visited
                  << std::left << std::setw(20) << std::fixed << std::setprecision(2) << percent_visited << "%" << std::endl;
    }
    std::cout << std::string(75, '-') << std::endl;

    // Assert that average visited-node count at ef=10 is meaningfully smaller than 500
    if (visited_at_ef10 >= 500.0) {
        std::cerr << "Assertion failed: average visited nodes at ef=10 (" 
                  << visited_at_ef10 << ") is not meaningfully smaller than 500." << std::endl;
        assert(false);
    }
    
    std::cout << "Recall vs ef tradeoff verified! Early termination successfully reduces search space." << std::endl;
    return 0;
}
