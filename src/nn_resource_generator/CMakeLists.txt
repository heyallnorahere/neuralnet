cmake_minimum_required(VERSION 3.20)

add_executable(nn_resource_generator main.cpp)
set_target_properties(nn_resource_generator PROPERTIES
    CXX_STANDARD 20
    FOLDER "neuralnet")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_link_libraries(nn_resource_generator PRIVATE stdc++fs)
endif()