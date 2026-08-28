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
