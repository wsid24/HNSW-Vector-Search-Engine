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
        return l2_squared_distance_best(query, node_vec, dim_);
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
            if (layer >= neighbor_counts_[id_c].size()) continue; 
            
            const uint32_t* neighbors = neighbor_slots(id_c, layer);
            uint16_t num_neighbors = get_neighbor_count(id_c, layer);
            
            for (uint16_t i = 0; i < num_neighbors; ++i) {
                uint32_t neighbor_id = neighbors[i];
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
        return l2_squared_distance_best(vec1, vec2, dim_);
    }

    void insert(const float* vector) {
        uint32_t vec_id = vector_store_.add_vector(vector);
        int new_node_layer = assign_random_layer();
        
        GraphNode new_node;
        new_node.vector_id = vec_id;
        new_node.max_layer = new_node_layer;
        
        uint32_t new_node_id = allocate_node_in_arena(new_node_layer);
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
                if (layer <= nodes_[curr_ep].max_layer && layer < static_cast<int>(neighbor_counts_[curr_ep].size())) {
                    const uint32_t* neighbors = neighbor_slots(curr_ep, layer);
                    uint16_t num_neighbors = get_neighbor_count(curr_ep, layer);
                    for (uint16_t i = 0; i < num_neighbors; ++i) {
                        uint32_t neighbor = neighbors[i];
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
            
            uint32_t* slots_new = neighbor_slots(new_node_id, layer);
            for (size_t i = 0; i < selected.size(); ++i) {
                slots_new[i] = selected[i];
            }
            set_neighbor_count(new_node_id, layer, selected.size());
            
            // Connect bidirectionally and prune if necessary
            for (uint32_t neighbor_id : selected) {
                uint32_t* slots_n = neighbor_slots(neighbor_id, layer);
                uint16_t n_count = get_neighbor_count(neighbor_id, layer);
                
                slots_n[n_count] = new_node_id;
                n_count++;
                set_neighbor_count(neighbor_id, layer, n_count);
                
                if (n_count > M_max) {
                    // Gather all candidates (existing + new) and their distances to the neighbor
                    std::vector<std::pair<float, uint32_t>> neighbor_candidates;
                    neighbor_candidates.reserve(n_count);
                    
                    const float* neighbor_vec = vector_store_.get_vector(nodes_[neighbor_id].vector_id);
                    for (size_t i = 0; i < n_count; ++i) {
                        uint32_t n_id = slots_n[i];
                        float d = distance_to_query(neighbor_vec, n_id);
                        neighbor_candidates.push_back({d, n_id});
                    }
                    std::sort(neighbor_candidates.begin(), neighbor_candidates.end());
                    
                    auto pruned = select_neighbors_heuristic(neighbor_vec, neighbor_candidates, M_max);
                    for (size_t i = 0; i < pruned.size(); ++i) {
                        slots_n[i] = pruned[i];
                    }
                    set_neighbor_count(neighbor_id, layer, pruned.size());
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
                if (layer <= nodes_[curr_ep].max_layer && layer < static_cast<int>(neighbor_counts_[curr_ep].size())) {
                    const uint32_t* neighbors = neighbor_slots(curr_ep, layer);
                    uint16_t num_neighbors = get_neighbor_count(curr_ep, layer);
                    for (uint16_t i = 0; i < num_neighbors; ++i) {
                        uint32_t neighbor = neighbors[i];
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
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
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
        if (num_nodes > 0) {
            out.write(reinterpret_cast<const char*>(nodes_.data()), num_nodes * sizeof(GraphNode));
        }
        
        // Write arena
        size_t arena_sz = neighbor_arena_.size();
        out.write(reinterpret_cast<const char*>(&arena_sz), sizeof(arena_sz));
        if (arena_sz > 0) {
            out.write(reinterpret_cast<const char*>(neighbor_arena_.data()), arena_sz * sizeof(uint32_t));
        }
        
        size_t offset_sz = node_arena_offset_.size();
        out.write(reinterpret_cast<const char*>(&offset_sz), sizeof(offset_sz));
        if (offset_sz > 0) {
            out.write(reinterpret_cast<const char*>(node_arena_offset_.data()), offset_sz * sizeof(size_t));
        }
        
        size_t counts_sz = neighbor_counts_.size();
        out.write(reinterpret_cast<const char*>(&counts_sz), sizeof(counts_sz));
        for (const auto& counts : neighbor_counts_) {
            size_t c_sz = counts.size();
            out.write(reinterpret_cast<const char*>(&c_sz), sizeof(c_sz));
            if (c_sz > 0) {
                out.write(reinterpret_cast<const char*>(counts.data()), c_sz * sizeof(uint16_t));
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
            uint32_t version;
            in.read(reinterpret_cast<char*>(&version), sizeof(version));
            if (!in || version != 1) {
                throw std::runtime_error("Unsupported file version or corrupted file");
            }
            
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
            if (num_nodes > 0) {
                in.read(reinterpret_cast<char*>(nodes_.data()), num_nodes * sizeof(GraphNode));
            }
            
            // Read arena
            size_t arena_sz;
            in.read(reinterpret_cast<char*>(&arena_sz), sizeof(arena_sz));
            if (!in) throw std::runtime_error("EOF while reading arena size");
            neighbor_arena_.resize(arena_sz);
            if (arena_sz > 0) {
                in.read(reinterpret_cast<char*>(neighbor_arena_.data()), arena_sz * sizeof(uint32_t));
            }
            
            size_t offset_sz;
            in.read(reinterpret_cast<char*>(&offset_sz), sizeof(offset_sz));
            if (!in) throw std::runtime_error("EOF while reading arena offset size");
            node_arena_offset_.resize(offset_sz);
            if (offset_sz > 0) {
                in.read(reinterpret_cast<char*>(node_arena_offset_.data()), offset_sz * sizeof(size_t));
            }
            
            size_t counts_sz;
            in.read(reinterpret_cast<char*>(&counts_sz), sizeof(counts_sz));
            if (!in) throw std::runtime_error("EOF while reading neighbor counts size");
            neighbor_counts_.resize(counts_sz);
            for (size_t i = 0; i < counts_sz; ++i) {
                size_t c_sz;
                in.read(reinterpret_cast<char*>(&c_sz), sizeof(c_sz));
                if (!in) throw std::runtime_error("EOF while reading neighbor counts layers");
                neighbor_counts_[i].resize(c_sz);
                if (c_sz > 0) {
                    in.read(reinterpret_cast<char*>(neighbor_counts_[i].data()), c_sz * sizeof(uint16_t));
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
    size_t dim() const { return dim_; }
    size_t M() const { return M_; }
    size_t efConstruction() const { return efConstruction_; }

private:
    size_t dim_;
    size_t M_;
    size_t efConstruction_;
    double mL_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    
public: 
    
    // since insert() is not yet implemented.
    VectorStore vector_store_;
    std::vector<GraphNode> nodes_;
    int entry_point_id_ = -1;
    int max_layer_ = -1;
    
    std::vector<uint32_t> neighbor_arena_;
    std::vector<size_t> node_arena_offset_;
    std::vector<std::vector<uint16_t>> neighbor_counts_;

    uint32_t* neighbor_slots(uint32_t node_id, int layer) {
        size_t offset = node_arena_offset_[node_id];
        if (layer == 0) {
            return &neighbor_arena_[offset];
        } else {
            return &neighbor_arena_[offset + (2 * M_ + 1) + (layer - 1) * (M_ + 1)];
        }
    }

    const uint32_t* neighbor_slots(uint32_t node_id, int layer) const {
        size_t offset = node_arena_offset_[node_id];
        if (layer == 0) {
            return &neighbor_arena_[offset];
        } else {
            return &neighbor_arena_[offset + (2 * M_ + 1) + (layer - 1) * (M_ + 1)];
        }
    }

    uint16_t get_neighbor_count(uint32_t node_id, int layer) const {
        return neighbor_counts_[node_id][layer];
    }

    void set_neighbor_count(uint32_t node_id, int layer, uint16_t count) {
        neighbor_counts_[node_id][layer] = count;
    }
    
    
    uint32_t allocate_node_in_arena(int node_layer) {
        uint32_t node_id = nodes_.size();
        size_t offset = neighbor_arena_.size();
        node_arena_offset_.push_back(offset);
        
        // Give each layer 1 extra slot to hold the temporary new edge before pruning
        size_t slot_count = (2 * M_ + 1) + node_layer * (M_ + 1);
        neighbor_arena_.resize(offset + slot_count, UINT32_MAX);
        
        neighbor_counts_.push_back(std::vector<uint16_t>(node_layer + 1, 0));
        return node_id;
    }
};

} // namespace hnsw
