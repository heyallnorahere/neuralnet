cmake_minimum_required(VERSION 3.21.0)
set(NN_DEFS PUBLIC $<$<CONFIG:Debug>:NN_DEBUG>)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND NN_LIBRARIES pthread stdc++fs)
endif()

if(BUILD_SHARED_LIBS)
    set(NN_LIB_TYPE SHARED)
    list(APPEND NN_DEFS PUBLIC NN_SHARED PRIVATE NN_EXPORT)
else()
    set(NN_LIB_TYPE STATIC)
endif()

file(GLOB_RECURSE NN_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_library(neuralnet STATIC ${NN_SOURCE})

target_link_libraries(neuralnet PUBLIC ${NN_LIBRARIES})
target_include_directories(neuralnet PUBLIC ${NN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(neuralnet ${NN_DEFS})
target_precompile_headers(neuralnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/nnpch.h")
set_target_properties(neuralnet PROPERTIES CXX_STANDARD 17)