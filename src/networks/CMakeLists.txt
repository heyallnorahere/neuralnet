cmake_minimum_required(VERSION 3.21.0)

set(NETWORK_LIBRARIES neuralnet)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND NETWORK_LIBRARIES pthread stdc++fs)
endif()

if(NN_BUILD_DEBUG_GUI)
    list(APPEND NETWORK_DEFS NN_BUILD_DEBUG_GUI)
    list(APPEND NETWORK_LIBRARIES glfw)
endif()

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h")
add_library(nn_common STATIC ${COMMON_SOURCES})
target_link_libraries(nn_common PUBLIC ${NETWORK_LIBRARIES})
target_compile_definitions(nn_common PUBLIC ${NETWORK_DEFS})

file(GLOB NETWORK_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(NETWORK_SOURCE ${NETWORK_SOURCES})
    cmake_path(GET NETWORK_SOURCE FILENAME SOURCE_FILENAME)
    cmake_path(GET SOURCE_FILENAME STEM NETWORK_NAME)
    list(APPEND NETWORKS ${NETWORK_NAME})

    add_executable(${NETWORK_NAME} ${NETWORK_SOURCE})
    target_link_libraries(${NETWORK_NAME} PRIVATE nn_common)
endforeach()

set_target_properties(${NETWORKS} nn_common PROPERTIES
    CXX_STANDARD 20
    FOLDER "networks")