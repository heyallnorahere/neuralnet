cmake_minimum_required(VERSION 3.21.0)

add_subdirectory("nn_resource_generator")
add_subdirectory("neuralnet")

if(NN_BUILD_NETWORKS)
    add_subdirectory("networks")
endif()