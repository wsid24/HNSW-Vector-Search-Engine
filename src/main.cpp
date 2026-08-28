#include "hnsw/graph_node.hpp"
#include "hnsw/distance.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace hnsw;

struct Candidate {
    uint32_t id;
    float dist;
    
    // Sort so smaller distance comes first
    bool operator<(const Candidate& other) const {
        return dist < other.dist;
    }
};

int main() {
    const size_t dim = 128;
    const size_t num_vectors = 1000;
    const size_t num_queries = 10;
    const size_t k = 10; // top-k nearest neighbors

    std::cout << "Generating " << num_vectors << " vectors of dimension " << dim << "..." << std::endl;

    // 1. Generate 1000 random vectors
    std::mt19937 rng(42); // fixed seed
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    VectorStore store(dim, num_vectors);
    
    std::vector<float> temp_vec(dim);
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            temp_vec[d] = dist(rng);
        }
        store.add_vector(temp_vec.data());
    }

    // 2. Generate 10 query vectors using the same RNG stream
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            queries[i][d] = dist(rng);
        }
    }

    // 3 & 4. Compute brute force distances, find top 10, print, and save to file
    std::string out_path = std::string(PROJECT_ROOT_DIR) + "/data/ground_truth.txt";
    std::ofstream out_file(out_path);
    if (!out_file) {
        std::cerr << "Failed to open " << out_path << " for writing." << std::endl;
        return 1;
    }

    std::cout << "Running brute-force search for " << num_queries << " queries..." << std::endl;
    
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<Candidate> candidates;
        candidates.reserve(num_vectors);

        for (uint32_t i = 0; i < store.size(); ++i) {
            float d = l2_squared_distance(queries[q].data(), store.get_vector(i), dim);
            candidates.push_back({i, d});
        }

        // Sort to get the closest `k`
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());

        // Print to console
        std::cout << "Query " << q << " Top-" << k << " NNs: ";
        for (size_t i = 0; i < k; ++i) {
            std::cout << "[id=" << candidates[i].id << ", d=" << std::fixed << std::setprecision(4) << candidates[i].dist << "] ";
        }
        std::cout << std::endl;

        // Write to file format: query_id: id1:dist1, id2:dist2, ...
        out_file << q << ": ";
        for (size_t i = 0; i < k; ++i) {
            out_file << candidates[i].id << ":" << candidates[i].dist;
            if (i < k - 1) out_file << ", ";
        }
        out_file << "\n";
    }

    out_file.close();
    std::cout << "Ground truth saved to " << out_path << std::endl;

    return 0;
}
