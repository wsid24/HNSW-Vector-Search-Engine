#include "httplib.h"
#include "json.hpp"
#include "hnsw/hnsw_index.hpp"
#include <iostream>
#include <vector>
#include <map>

using json = nlohmann::json;
using namespace hnsw;

int main() {
    std::cout << "Loading HNSW index from data/index_99k.bin..." << std::endl;
    HNSWIndex index(384, 16, 200, 42); // Dummy params, will be overwritten by load
    try {
        index.load("data/index_99k.bin");
    } catch (const std::exception& e) {
        std::cerr << "Failed to load index: " << e.what() << std::endl;
        return 1;
    }
    
    // Stats
    std::map<int, int> layer_histogram;
    for (const auto& node : index.nodes_) {
        layer_histogram[node.max_layer]++;
    }
    
    std::cout << "Index loaded successfully." << std::endl;
    std::cout << "Nodes: " << index.nodes_.size() << ", Dim: " << index.dim() << std::endl;

    httplib::Server svr;

    svr.Get("/stats", [&index, layer_histogram](const httplib::Request& req, httplib::Response& res) {
        json j;
        j["node_count"] = index.nodes_.size();
        j["dim"] = index.dim();
        j["M"] = index.M();
        j["efConstruction"] = index.efConstruction();
        
        json hist;
        for (const auto& [layer, count] : layer_histogram) {
            hist[std::to_string(layer)] = count;
        }
        j["layer_histogram"] = hist;
        
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/search", [&index](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            
            if (!j.contains("query") || !j["query"].is_array()) {
                res.status = 400;
                res.set_content(json{{"error", "Missing or invalid 'query' array."}}.dump(), "application/json");
                return;
            }
            
            auto query_array = j["query"];
            if (query_array.size() != index.dim()) {
                res.status = 400;
                res.set_content(json{{"error", "Query dimension mismatch. Expected " + std::to_string(index.dim())}}.dump(), "application/json");
                return;
            }
            
            if (!j.contains("k") || !j["k"].is_number()) {
                res.status = 400;
                res.set_content(json{{"error", "Missing or invalid 'k'."}}.dump(), "application/json");
                return;
            }
            
            if (!j.contains("ef") || !j["ef"].is_number()) {
                res.status = 400;
                res.set_content(json{{"error", "Missing or invalid 'ef'."}}.dump(), "application/json");
                return;
            }
            
            size_t k = j["k"].get<size_t>();
            size_t ef = j["ef"].get<size_t>();
            std::vector<float> query;
            query.reserve(index.dim());
            for (const auto& val : query_array) {
                query.push_back(val.get<float>());
            }
            
            auto search_results = index.search(query.data(), k, ef);
            
            json results_array = json::array();
            for (const auto& r : search_results) {
                results_array.push_back({{"id", r.second}, {"distance", r.first}});
            }
            
            json response;
            response["results"] = results_array;
            res.set_content(response.dump(), "application/json");
            
        } catch (const json::parse_error& e) {
            res.status = 400;
            res.set_content(json{{"error", "Invalid JSON body."}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json{{"error", std::string("Internal server error: ") + e.what()}}.dump(), "application/json");
        }
    });

    std::cout << "Starting HTTP server on port 8080..." << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}
