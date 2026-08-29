cmake_minimum_required(VERSION 3.10)
project(hnsw_vector_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# The core library
add_library(hnsw_core src/distance.cpp)

# The main executable
add_executable(main src/main.cpp)
target_link_libraries(main hnsw_core)
target_compile_definitions(main PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# Test executable for distances
add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance hnsw_core)


# Test executable for graph node
add_executable(test_graph_node tests/test_graph_node.cpp)

# Test executable for layer assignment
add_executable(test_layer_assignment tests/test_layer_assignment.cpp)

# Test executable for search layer
add_executable(test_search_layer tests/test_search_layer.cpp)
target_link_libraries(test_search_layer hnsw_core)

# Test executable for insert
add_executable(test_insert tests/test_insert.cpp)
target_link_libraries(test_insert hnsw_core)

# Test executable for search correctness
add_executable(test_search_correctness tests/test_search_correctness.cpp)
target_link_libraries(test_search_correctness hnsw_core)
target_compile_definitions(test_search_correctness PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# Test executable for recall vs ef
add_executable(test_recall_vs_ef tests/test_recall_vs_ef.cpp)
target_link_libraries(test_recall_vs_ef hnsw_core)
target_compile_definitions(test_recall_vs_ef PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# Benchmark executable for real dataset
add_executable(run_real_dataset benchmarks/run_real_dataset.cpp)
target_link_libraries(run_real_dataset hnsw_core)
target_compile_definitions(run_real_dataset PRIVATE PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
