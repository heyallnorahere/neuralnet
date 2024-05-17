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

if(NN_SUPPORT_VULKAN)
    list(APPEND NN_LIBRARIES volk vma)
endif()

set(NN_RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/resources")
file(GLOB_RECURSE NN_RESOURCES CONFIGURE_DEPENDS "${NN_RESOURCE_DIR}/*")
file(GLOB_RECURSE NN_SHADER_SOURCE CONFIGURE_DEPENDS "${NN_RESOURCE_DIR}/*.glsl")
list(REMOVE_ITEM NN_RESOURCES ${NN_SHADER_SOURCE})

set(NN_RESOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/resources.gen.cpp")
set(NN_RESOURCE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/resources.cpp.in")
set(NN_RESOURCE_DATA_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/resource_data.cpp.in")
list(APPEND NN_SOURCE ${NN_RESOURCE_FILE})

if(NN_SUPPORT_VULKAN)
    file(GLOB NN_COMPUTE_SHADERS CONFIGURE_DEPENDS "${NN_RESOURCE_DIR}/glsl/*.glsl")

    set(NN_SPIRV_DIR "${NN_RESOURCE_DIR}/spirv")
    make_directory(${NN_SPIRV_DIR})

    foreach(SHADER_SOURCE ${NN_COMPUTE_SHADERS})
        cmake_path(GET SHADER_SOURCE STEM SHADER_NAME)
        set(SPIRV_PATH "${NN_SPIRV_DIR}/${SHADER_NAME}.spv")

        add_custom_command(OUTPUT ${SPIRV_PATH} DEPENDS ${NN_SHADER_SOURCE} COMMAND glslc_exe -fshader-stage=comp ${SHADER_SOURCE} -O --target-env=vulkan1.0 -o ${SPIRV_PATH})

        list(FIND NN_RESOURCES ${SPIRV_PATH} PATH_INDEX)
        if(PATH_INDEX LESS 0)
            list(APPEND NN_RESOURCES ${SPIRV_PATH})
        endif()
    endforeach()
endif()

foreach(RESOURCE_PATH ${NN_RESOURCES})
    string(APPEND NN_RESOURCE_STRING ":${RESOURCE_PATH}")
endforeach()

set(NN_GENERATOR_COMMAND nn_resource_generator -o ${NN_RESOURCE_FILE} -t ${NN_RESOURCE_TEMPLATE} -d ${NN_RESOURCE_DATA_TEMPLATE} -i ${NN_RESOURCE_STRING} -r ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_command(OUTPUT ${NN_RESOURCE_FILE} COMMAND ${NN_GENERATOR_COMMAND} DEPENDS ${NN_RESOURCES} ${NN_RESOURCE_TEMPLATE} ${NN_RESOURCE_DATA_TEMPLATE})

add_library(neuralnet STATIC ${NN_SOURCE})

target_link_libraries(neuralnet PUBLIC ${NN_LIBRARIES})
target_include_directories(neuralnet PUBLIC ${NN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(neuralnet ${NN_DEFS})
target_precompile_headers(neuralnet PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/nnpch.h")
set_target_properties(neuralnet PROPERTIES
    CXX_STANDARD 20
    FOLDER "neuralnet")