cmake_minimum_required(VERSION 3.10)
project(HandwritingCNN)

set(CMAKE_CXX_STANDARD 17)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(HandwritingCNN ${SOURCES})
