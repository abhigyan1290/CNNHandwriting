cmake_minimum_required(VERSION 3.10)
project(HandwritingCNN)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (installed via package manager)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find OpenCV (if using)
option(USE_OPENCV "Use OpenCV for image processing" OFF)
if(USE_OPENCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Include directories for your own headers
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# Add all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Define the executable
add_executable(HandwritingCNN ${SOURCES})

# Link Eigen3 to the executable
target_link_libraries(HandwritingCNN PRIVATE Eigen3::Eigen)

# Link OpenCV libraries if used
if(USE_OPENCV)
    target_link_libraries(HandwritingCNN PRIVATE ${OpenCV_LIBS})
endif()
