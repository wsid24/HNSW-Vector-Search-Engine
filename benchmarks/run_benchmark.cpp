#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
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

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

int main() {
    const size_t dim = 384;
    const size_t num_db = 99000;
    const size_t num_queries = 1000;
    const size_t k = 10;
    
    std::string root_dir = std::string(PROJECT_ROOT_DIR);
    std::string db_path = root_dir + "/data/embeddings.bin";
    std::string q_path = root_dir + "/data/queries.bin";
    std::string index_path = root_dir + "/data/index_99k.bin";
    std::string gt_path = root_dir + "/data/ground_truth_full.txt";
    std::string csv_path = root_dir + "/benchmarks/results_hnsw.csv";
    
    std::cout << "Loading dataset..." << std::endl;
    std::vector<float> db_floats = load_raw_floats(db_path);
    std::vector<float> query_floats = load_raw_floats(q_path);
    
    if (db_floats.size() != num_db * dim) {
        throw std::runtime_error("embeddings.bin size mismatch");
    }
    if (query_floats.size() != num_queries * dim) {
        throw std::runtime_error("queries.bin size mismatch");
    }
    
    HNSWIndex index(dim, 16, 200, 42);
    
    if (file_exists(index_path)) {
        index.load(root_dir + "/data/index_99k.bin");
        std::cout << "Found cached index at " << root_dir + "/data/index_99k.bin" << ", loading..." << std::endl;
    } else {
        std::cout << "Building HNSW index from scratch..." << std::endl;
        auto t_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < num_db; ++i) {
            index.insert(db_floats.data() + i * dim);
            if ((i + 1) % 10000 == 0) {
                std::cout << "Inserted " << (i + 1) << " / " << num_db << " vectors..." << std::endl;
            }
        }
        auto t_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> build_time = t_end - t_start;
        std::cout << "Index built in " << build_time.count() << " seconds." << std::endl;
        
        std::cout << "Saving index to " << index_path << "..." << std::endl;
        index.save(index_path);
    }
    
    std::vector<std::unordered_set<uint32_t>> ground_truth(num_queries);
    
    std::cout << "Generating brute-force ground truth for " << num_queries << " queries..." << std::endl;
    auto gt_start = std::chrono::steady_clock::now();
    
    std::ofstream gt_out(gt_path);
    if (!gt_out) {
        throw std::runtime_error("Failed to open ground truth output file: " + gt_path);
    }
    
    for (size_t q = 0; q < num_queries; ++q) {
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
    
    std::ofstream csv_out(csv_path);
    if (!csv_out) {
        throw std::runtime_error("Failed to open csv output file: " + csv_path);
    }
    csv_out << "ef,avg_recall@10,qps,p50_ms,p95_ms,p99_ms\n";
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(10) << "ef"
              << std::left << std::setw(15) << "Recall@10"
              << std::left << std::setw(15) << "QPS"
              << std::left << std::setw(10) << "p50 (ms)"
              << std::left << std::setw(10) << "p95 (ms)"
              << std::left << std::setw(10) << "p99 (ms)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (size_t ef : ef_values) {
        double total_recall = 0.0;
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        
        double total_time_s = 0.0;
        
        for (size_t q = 0; q < num_queries; ++q) {
            const float* q_vec = query_floats.data() + q * dim;
            
            auto t0 = std::chrono::steady_clock::now();
            auto results = index.search(q_vec, k, ef);
            auto t1 = std::chrono::steady_clock::now();
            
            std::chrono::duration<double> dt = t1 - t0;
            double ms = dt.count() * 1000.0;
            latencies.push_back(ms);
            total_time_s += dt.count();
            
            int matches = 0;
            for (const auto& r : results) {
                if (ground_truth[q].find(r.second) != ground_truth[q].end()) {
                    matches++;
                }
            }
            total_recall += static_cast<double>(matches) / k;
        }
        
        double avg_recall = total_recall / num_queries;
        double qps = static_cast<double>(num_queries) / total_time_s;
        
        std::sort(latencies.begin(), latencies.end());
        double p50 = latencies[static_cast<size_t>(num_queries * 0.50)];
        double p95 = latencies[static_cast<size_t>(num_queries * 0.95)];
        double p99 = latencies[static_cast<size_t>(num_queries * 0.99)];
        
        std::cout << std::left << std::setw(10) << ef
                  << std::left << std::setw(15) << std::fixed << std::setprecision(4) << avg_recall
                  << std::left << std::setw(15) << std::fixed << std::setprecision(2) << qps
                  << std::left << std::setw(10) << std::fixed << std::setprecision(3) << p50
                  << std::left << std::setw(10) << std::fixed << std::setprecision(3) << p95
                  << std::left << std::setw(10) << std::fixed << std::setprecision(3) << p99 << std::endl;
                  
        csv_out << ef << "," << avg_recall << "," << qps << "," << p50 << "," << p95 << "," << p99 << "\n";
    }
    std::cout << std::string(70, '-') << std::endl;
    csv_out.close();
    
    return 0;
}
