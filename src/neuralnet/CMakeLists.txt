cmake_minimum_required(VERSION 3.21.0)

set(NN_DEFS PUBLIC $<$<CONFIG:Debug>:NN_DEBUG>)
set(NN_LIBRARIES Tracy::TracyClient nlohmann_json)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND NN_LIBRARIES pthread stdc++fs)
endif()

if(BUILD_SHARED_LIBS)
    set(NN_LIB_TYPE SHARED)
    list(APPEND NN_DEFS PUBLIC NN_SHARED PRIVATE NN_EXPORT)
    list(APPEND NN_LIBRARIES zlib)
else()
    set(NN_LIB_TYPE STATIC)
    list(APPEND NN_LIBRARIES zlibstatic)
endif()

file(GLOB_RECURSE NN_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB NN_ALL_EVALUATORS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/evaluators/*_evaluator.cpp")
list(REMOVE_ITEM NN_SOURCE ${NN_ALL_EVALUATORS})

foreach(EVALUATOR_NAME ${NN_EVALUATORS})
    list(APPEND NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/evaluators/${EVALUATOR_NAME}_evaluator.cpp")
    list(APPEND NN_DEFS PUBLIC NN_SUPPORT_${EVALUATOR_NAME})
endforeach()

add_library(neuralnet STATIC ${NN_SOURCE})

target_link_libraries(neuralnet PUBLIC ${NN_LIBRARIES})
target_include_directories(neuralnet PUBLIC ${NN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(neuralnet ${NN_DEFS})
target_precompile_headers(neuralnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/nnpch.h")
set_target_properties(neuralnet PROPERTIES CXX_STANDARD 20)