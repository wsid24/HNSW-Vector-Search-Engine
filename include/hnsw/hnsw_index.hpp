#pragma once

#include "hnsw/graph_node.hpp"
#include "hnsw/distance.hpp"
#include <cmath>
#include <random>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <functional>
#include <fstream>
#include <stdexcept>

namespace hnsw {

class HNSWIndex {
public:
    // Constructor
    explicit HNSWIndex(size_t dim, size_t M = 16, size_t efConstruction = 200, uint32_t seed = 42)
        : dim_(dim),
          M_(M),
          efConstruction_(efConstruction),
          mL_(1.0 / std::log(static_cast<double>(M))),
          rng_(seed),
          uniform_dist_(0.0, 1.0),
          vector_store_(dim) {
    }

    // Assigns a random layer based on the exponentially decaying distribution
    int assign_random_layer() {
        double u = uniform_dist_(rng_);
        while (u == 0.0) {
            u = uniform_dist_(rng_); // Guard against -ln(0)
        }
        
        // Compute level = floor(-ln(u) * mL_)
        int level = static_cast<int>(std::floor(-std::log(u) * mL_));
        return level;
    }

    // Computes distance from query vector to a specific node in the graph
    float distance_to_query(const float* query, uint32_t node_id) const {
        const float* node_vec = vector_store_.get_vector(nodes_[node_id].vector_id);
        return l2_squared_distance(query, node_vec, dim_);
    }

    // Algorithm 2: Greedy search bounded to ef results at a specific layer
    std::vector<std::pair<float, uint32_t>> search_layer(
        const float* query, 
        const std::vector<uint32_t>& entry_points, 
        size_t ef, 
        int layer,
        size_t* out_visited_count = nullptr) const 
    {
        using Pair = std::pair<float, uint32_t>;
        
        // Min-heap for candidates: top is the closest (smallest distance)
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> candidates;
        
        // Max-heap for results: top is the furthest (largest distance) among the closest found
        std::priority_queue<Pair> results;
        
        std::unordered_set<uint32_t> visited;
        
        for (uint32_t ep : entry_points) {
            float dist = distance_to_query(query, ep);
            candidates.push({dist, ep});
            results.push({dist, ep});
            visited.insert(ep);
        }
        
        while (!candidates.empty()) {
            Pair c = candidates.top();
            candidates.pop();
            
            float dist_c = c.first;
            uint32_t id_c = c.second;
            
            // Deliberate deviation from the paper's Algorithm 2: the paper breaks purely on
            // distance(c) > distance(f), but we additionally require results.size() >= ef before breaking,
            // to guarantee the result set is filled to ef before allowing early exit — trading a small
            // amount of extra exploration early in the search for a correctness guarantee.
            if (results.size() >= ef && dist_c > results.top().first) {
                break;
            }
            
            // Guard against layer out of bounds (can happen if node doesn't reach this layer)
            if (layer > nodes_[id_c].max_layer) continue;
            if (layer >= nodes_[id_c].neighbors.size()) continue; 
            
            const auto& neighbors = nodes_[id_c].neighbors[layer];
            for (uint32_t neighbor_id : neighbors) {
                if (visited.find(neighbor_id) == visited.end()) {
                    visited.insert(neighbor_id);
                    float dist_e = distance_to_query(query, neighbor_id);
                    
                    if (results.size() < ef || dist_e < results.top().first) {
                        candidates.push({dist_e, neighbor_id});
                        results.push({dist_e, neighbor_id});
                        
                        if (results.size() > ef) {
                            results.pop(); // Evict the worst
                        }
                    }
                }
            }
        }
        
        std::vector<Pair> final_results;
        final_results.reserve(results.size());
        while (!results.empty()) {
            final_results.push_back(results.top());
            results.pop();
        }
        
        // Results came out of a max-heap, so they are descending distance.
        // Reverse them to be ascending (closest first).
        std::reverse(final_results.begin(), final_results.end());
        
        if (out_visited_count) {
            *out_visited_count = visited.size();
        }
        
        return final_results;
    }
    // Heuristic for selecting neighbors (Algorithm 4)
    std::vector<uint32_t> select_neighbors_heuristic(
        const float* query,
        const std::vector<std::pair<float, uint32_t>>& candidates,
        size_t M) const 
    {
        std::vector<uint32_t> result;
        result.reserve(M);
        
        for (const auto& c : candidates) {
            if (result.size() >= M) break;
            
            float dist_to_query = c.first;
            uint32_t candidate_id = c.second;
            
            bool good = true;
            for (uint32_t r_id : result) {
                float dist_to_r = distance_between_nodes(candidate_id, r_id);
                // If candidate is closer to an already selected neighbor than to the query, skip it
                if (dist_to_r < dist_to_query) {
                    good = false;
                    break;
                }
            }
            
            if (good) {
                result.push_back(candidate_id);
            }
        }
        
        return result;
    }
    
    // Computes distance between two nodes in the graph
    float distance_between_nodes(uint32_t id1, uint32_t id2) const {
        const float* vec1 = vector_store_.get_vector(nodes_[id1].vector_id);
        const float* vec2 = vector_store_.get_vector(nodes_[id2].vector_id);
        return l2_squared_distance(vec1, vec2, dim_);
    }

    void insert(const float* vector) {
        uint32_t vec_id = vector_store_.add_vector(vector);
        int new_node_layer = assign_random_layer();
        
        GraphNode new_node;
        new_node.vector_id = vec_id;
        new_node.max_layer = new_node_layer;
        new_node.neighbors.resize(new_node_layer + 1);
        
        uint32_t new_node_id = nodes_.size();
        nodes_.push_back(new_node);
        
        if (entry_point_id_ == -1) {
            entry_point_id_ = new_node_id;
            max_layer_ = new_node_layer;
            return;
        }
        
        int curr_ep = entry_point_id_;
        float curr_dist = distance_to_query(vector, curr_ep);
        
        // Greedily descend through layers strictly above new_node_layer
        for (int layer = max_layer_; layer > new_node_layer; --layer) {
            bool changed = true;
            while (changed) {
                changed = false;
                if (layer <= nodes_[curr_ep].max_layer && layer < static_cast<int>(nodes_[curr_ep].neighbors.size())) {
                    for (uint32_t neighbor : nodes_[curr_ep].neighbors[layer]) {
                        float dist = distance_to_query(vector, neighbor);
                        if (dist < curr_dist) {
                            curr_dist = dist;
                            curr_ep = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }
        
        std::vector<uint32_t> entry_points = { static_cast<uint32_t>(curr_ep) };
        
        // Connect and descend from min(max_layer_, new_node_layer) down to 0
        int start_layer = std::min(max_layer_, new_node_layer);
        for (int layer = start_layer; layer >= 0; --layer) {
            auto candidates = search_layer(vector, entry_points, efConstruction_, layer);
            
            size_t M_max = (layer == 0) ? (M_ * 2) : M_;
            std::vector<uint32_t> selected = select_neighbors_heuristic(vector, candidates, M_max);
            
            nodes_[new_node_id].neighbors[layer] = selected;
            
            // Connect bidirectionally and prune if necessary
            for (uint32_t neighbor_id : selected) {
                auto& neighbor_edges = nodes_[neighbor_id].neighbors[layer];
                neighbor_edges.push_back(new_node_id);
                
                if (neighbor_edges.size() > M_max) {
                    // Gather all candidates (existing + new) and their distances to the neighbor
                    std::vector<std::pair<float, uint32_t>> neighbor_candidates;
                    neighbor_candidates.reserve(neighbor_edges.size());
                    
                    const float* neighbor_vec = vector_store_.get_vector(nodes_[neighbor_id].vector_id);
                    for (uint32_t n_id : neighbor_edges) {
                        float d = distance_to_query(neighbor_vec, n_id);
                        neighbor_candidates.push_back({d, n_id});
                    }
                    std::sort(neighbor_candidates.begin(), neighbor_candidates.end());
                    
                    nodes_[neighbor_id].neighbors[layer] = select_neighbors_heuristic(neighbor_vec, neighbor_candidates, M_max);
                }
            }
            
            // Entry points for the next layer down are the candidates found at this layer
            entry_points.clear();
            entry_points.reserve(candidates.size());
            for (const auto& c : candidates) {
                entry_points.push_back(c.second);
            }
        }
        
        if (new_node_layer > max_layer_) {
            max_layer_ = new_node_layer;
            entry_point_id_ = new_node_id;
        }
    }
    
    // Top-k search algorithm
    std::vector<std::pair<float, uint32_t>> search(
        const float* query, 
        size_t k, 
        size_t ef, 
        size_t* out_visited_count = nullptr) const 
    {
        if (entry_point_id_ == -1) {
            if (out_visited_count) *out_visited_count = 0;
            return {};
        }
        
        if (ef < k) {
            ef = k;
        }
        
        int curr_ep = entry_point_id_;
        float curr_dist = distance_to_query(query, curr_ep);
        
        // Descend to layer 1 using greedy search (ef=1 equivalent)
        for (int layer = max_layer_; layer >= 1; --layer) {
            bool changed = true;
            while (changed) {
                changed = false;
                if (layer <= nodes_[curr_ep].max_layer && layer < static_cast<int>(nodes_[curr_ep].neighbors.size())) {
                    for (uint32_t neighbor : nodes_[curr_ep].neighbors[layer]) {
                        float dist = distance_to_query(query, neighbor);
                        if (dist < curr_dist) {
                            curr_dist = dist;
                            curr_ep = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }
        
        // Search at layer 0 with provided ef
        std::vector<uint32_t> entry_points = { static_cast<uint32_t>(curr_ep) };
        auto results = search_layer(query, entry_points, ef, 0, out_visited_count);
        
        if (results.size() > k) {
            results.resize(k);
        }
        
        return results;
    }
    
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for saving: " + path);
        }
        
        // Write header
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
        out.write(reinterpret_cast<const char*>(&efConstruction_), sizeof(efConstruction_));
        out.write(reinterpret_cast<const char*>(&entry_point_id_), sizeof(entry_point_id_));
        out.write(reinterpret_cast<const char*>(&max_layer_), sizeof(max_layer_));
        
        // Write vector store
        size_t num_vectors = vector_store_.size();
        out.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
        if (num_vectors > 0) {
            out.write(reinterpret_cast<const char*>(vector_store_.raw_data()), num_vectors * dim_ * sizeof(float));
        }
        
        // Write nodes
        size_t num_nodes = nodes_.size();
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
        
        for (const auto& node : nodes_) {
            out.write(reinterpret_cast<const char*>(&node.vector_id), sizeof(node.vector_id));
            out.write(reinterpret_cast<const char*>(&node.max_layer), sizeof(node.max_layer));
            
            size_t num_layers = node.neighbors.size();
            out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
            
            for (size_t l = 0; l < num_layers; ++l) {
                size_t num_neighbors = node.neighbors[l].size();
                out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));
                if (num_neighbors > 0) {
                    out.write(reinterpret_cast<const char*>(node.neighbors[l].data()), num_neighbors * sizeof(uint32_t));
                }
            }
        }
        
        if (!out) {
            throw std::runtime_error("Error occurred while writing to file: " + path);
        }
    }
    
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open file for loading: " + path);
        }
        
        try {
            // Read header
            in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
            in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
            in.read(reinterpret_cast<char*>(&efConstruction_), sizeof(efConstruction_));
            in.read(reinterpret_cast<char*>(&entry_point_id_), sizeof(entry_point_id_));
            in.read(reinterpret_cast<char*>(&max_layer_), sizeof(max_layer_));
            
            // Read vector store
            size_t num_vectors;
            in.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
            if (!in) throw std::runtime_error("EOF while reading vector store size");
            
            vector_store_ = VectorStore(dim_, num_vectors);
            vector_store_.resize(num_vectors);
            if (num_vectors > 0) {
                in.read(reinterpret_cast<char*>(vector_store_.mutable_data()), num_vectors * dim_ * sizeof(float));
            }
            
            // Read nodes
            size_t num_nodes;
            in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
            if (!in) throw std::runtime_error("EOF while reading node count");
            
            nodes_.resize(num_nodes);
            
            for (size_t i = 0; i < num_nodes; ++i) {
                GraphNode& node = nodes_[i];
                in.read(reinterpret_cast<char*>(&node.vector_id), sizeof(node.vector_id));
                in.read(reinterpret_cast<char*>(&node.max_layer), sizeof(node.max_layer));
                
                size_t num_layers;
                in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
                if (!in) throw std::runtime_error("EOF while reading layers");
                
                node.neighbors.resize(num_layers);
                
                for (size_t l = 0; l < num_layers; ++l) {
                    size_t num_neighbors;
                    in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
                    if (num_neighbors > 0) {
                        node.neighbors[l].resize(num_neighbors);
                        in.read(reinterpret_cast<char*>(node.neighbors[l].data()), num_neighbors * sizeof(uint32_t));
                    }
                }
            }
            
            if (!in) {
                throw std::runtime_error("Error occurred while reading from file or unexpected EOF");
            }
            
            // Re-calculate derived values
            mL_ = 1.0 / std::log(static_cast<double>(M_));
            
        } catch (const std::exception& e) {
            throw std::runtime_error("File format corrupted or invalid data: " + std::string(e.what()));
        }
    }


private:
    size_t dim_;
    size_t M_;
    size_t efConstruction_;
    double mL_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    
public: 
    // Public temporarily to allow manual graph building in test_search_layer.cpp
    // since insert() is not yet implemented.
    VectorStore vector_store_;
    std::vector<GraphNode> nodes_;
    int entry_point_id_ = -1;
    int max_layer_ = -1;
};

} // namespace hnsw
