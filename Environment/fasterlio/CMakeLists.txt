cmake_minimum_required(VERSION 3.5)
project(faster_lio)

set(DEFAULT_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread  -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread  -fopenmp -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ")

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
# add_subdirectory(thirdparty/livox_ros_driver)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
# include(cmake/packages.cmake)

# options
option(WITH_IVOX_NODE_TYPE_PHC "Use PHC instead of default ivox node" OFF)

if (WITH_IVOX_NODE_TYPE_PHC)
    message("USING_IVOX_NODE_TYPE_PHC")
    add_definitions(-DIVOX_NODE_TYPE_PHC)
else ()
    message("USING_IVOX_NODE_TYPE_DEFAULT")
endif()

add_subdirectory(src)
add_subdirectory(app)
