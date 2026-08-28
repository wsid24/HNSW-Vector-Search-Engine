#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace hnsw {

class VectorStore {
public:
    explicit VectorStore(size_t dim, size_t expected_capacity = 0) : dim_(dim) {
        if (expected_capacity > 0) {
            data_.reserve(expected_capacity * dim);
        }
    }

    // Appends a vector to the contiguous storage and returns its assigned ID.
    // The data array must have length equal to dim_.
    uint32_t add_vector(const float* data) {
        uint32_t id = size();
        data_.insert(data_.end(), data, data + dim_);
        return id;
    }

    // Returns a pointer to the vector data for a given ID.
    // Note: this pointer is only valid as long as no reallocation happens in data_,
    // or if capacity has been reserved ahead of time.
    const float* get_vector(uint32_t id) const {
        if (id >= size()) {
            throw std::out_of_range("Vector ID " + std::to_string(id) + 
                                    " out of range (size: " + std::to_string(size()) + ")");
        }
        return data_.data() + (id * dim_);
    }

    // Returns the number of vectors stored.
    size_t size() const {
        return data_.size() / dim_;
    }
    
    // Returns the dimensionality of vectors in this store.
    size_t dim() const {
        return dim_;
    }

private:
    size_t dim_;
    std::vector<float> data_;
};

struct GraphNode {
    uint32_t vector_id;
    int max_layer;
    std::vector<std::vector<uint32_t>> neighbors; // outer index = layer, inner = neighbor IDs
};

} // namespace hnsw
