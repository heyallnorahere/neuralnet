cmake_minimum_required(VERSION 3.21.0)

file(GLOB_RECURSE NN_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_library(neuralnet STATIC ${NN_SOURCE})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND NN_LIBRARIES pthread stdc++fs)
endif()

target_link_libraries(neuralnet PUBLIC ${NN_LIBRARIES})
target_include_directories(neuralnet PUBLIC ${NN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_precompile_headers(neuralnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/nnpch.h")
set_target_properties(neuralnet PROPERTIES CXX_STANDARD 17)