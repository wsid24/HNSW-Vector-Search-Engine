#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include <iomanip>

using namespace hnsw;

std::vector<float> load_raw_floats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (size % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of sizeof(float)");
    }
    
    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    } else {
        throw std::runtime_error("Failed to read file completely");
    }
}

struct Candidate {
    uint32_t id;
    float dist;
    bool operator<(const Candidate& other) const {
        return dist < other.dist;
    }
};

int main() {
    const size_t dim = 384;
    const size_t num_db = 99000;
    const size_t num_queries = 1000;
    const size_t test_queries = 20;
    const size_t k = 10;
    
    std::string db_path = std::string(PROJECT_ROOT_DIR) + "/data/embeddings.bin";
    std::string q_path = std::string(PROJECT_ROOT_DIR) + "/data/queries.bin";
    
    std::cout << "Loading real dataset..." << std::endl;
    std::vector<float> db_floats = load_raw_floats(db_path);
    std::vector<float> query_floats = load_raw_floats(q_path);
    
    if (db_floats.size() != num_db * dim) {
        throw std::runtime_error("embeddings.bin size mismatch: expected " + std::to_string(num_db * dim) + " floats, got " + std::to_string(db_floats.size()));
    }
    
    if (query_floats.size() != num_queries * dim) {
        throw std::runtime_error("queries.bin size mismatch: expected " + std::to_string(num_queries * dim) + " floats, got " + std::to_string(query_floats.size()));
    }
    
    std::cout << "Building HNSW index..." << std::endl;
    HNSWIndex index(dim, 16, 200, 42); // M=16, efConstruction=200, seed=42
    
    auto t_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < num_db; ++i) {
        index.insert(db_floats.data() + i * dim);
        if ((i + 1) % 10000 == 0) {
            std::cout << "Inserted " << (i + 1) << " / " << num_db << " vectors..." << std::endl;
        }
    }
    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> build_time = t_end - t_start;
    
    std::cout << "Total build time: " << build_time.count() << " seconds" << std::endl;
    std::cout << "Average time per insert: " << (build_time.count() / num_db) * 1000.0 << " ms" << std::endl;
    
    std::cout << "Generating brute-force ground truth for " << test_queries << " queries..." << std::endl;
    std::vector<std::unordered_set<uint32_t>> ground_truth(test_queries);
    
    std::string gt_path = std::string(PROJECT_ROOT_DIR) + "/data/ground_truth_real.txt";
    std::ofstream gt_out(gt_path);
    if (!gt_out) {
        std::cerr << "Failed to open ground truth output file" << std::endl;
        return 1;
    }
    
    auto gt_start = std::chrono::steady_clock::now();
    for (size_t q = 0; q < test_queries; ++q) {
        const float* q_vec = query_floats.data() + q * dim;
        std::vector<Candidate> candidates;
        candidates.reserve(num_db);
        
        for (uint32_t i = 0; i < num_db; ++i) {
            float d = l2_squared_distance(q_vec, db_floats.data() + i * dim, dim);
            candidates.push_back({i, d});
        }
        
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
        
        gt_out << q << ": ";
        for (size_t i = 0; i < k; ++i) {
            ground_truth[q].insert(candidates[i].id);
            gt_out << candidates[i].id << ":" << candidates[i].dist;
            if (i < k - 1) gt_out << ", ";
        }
        gt_out << "\n";
    }
    auto gt_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> gt_time = gt_end - gt_start;
    gt_out.close();
    
    std::cout << "Ground truth generation took " << gt_time.count() << " seconds." << std::endl;
    
    std::vector<size_t> ef_values = {10, 20, 50, 100, 200};
    
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left << std::setw(10) << "ef"
              << std::left << std::setw(20) << "Avg Recall@10"
              << std::left << std::setw(25) << "Avg Nodes Visited"
              << std::left << std::setw(20) << "% of Graph Visited" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (size_t ef : ef_values) {
        double total_recall = 0.0;
        size_t total_visited = 0;
        
        for (size_t q = 0; q < test_queries; ++q) {
            const float* q_vec = query_floats.data() + q * dim;
            size_t visited_count = 0;
            auto results = index.search(q_vec, k, ef, &visited_count);
            
            int matches = 0;
            for (const auto& r : results) {
                if (ground_truth[q].find(r.second) != ground_truth[q].end()) {
                    matches++;
                }
            }
            
            total_recall += static_cast<double>(matches) / k;
            total_visited += visited_count;
        }
        
        double avg_recall = total_recall / test_queries;
        double avg_visited = static_cast<double>(total_visited) / test_queries;
        double percent_visited = (avg_visited / num_db) * 100.0;
        
        std::cout << std::left << std::setw(10) << ef
                  << std::left << std::setw(20) << std::fixed << std::setprecision(4) << avg_recall
                  << std::left << std::setw(25) << std::fixed << std::setprecision(1) << avg_visited
                  << std::left << std::setw(20) << std::fixed << std::setprecision(2) << percent_visited << "%" << std::endl;
    }
    std::cout << std::string(75, '-') << std::endl;
    
    return 0;
}
