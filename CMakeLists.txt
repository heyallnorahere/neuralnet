cmake_minimum_required(VERSION 3.21.0)
project(neuralnet)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(VULKAN_UNSUPPORTED_PLATFORMS Darwin iOS)
list(FIND VULKAN_UNSUPPORTED_PLATFORMS ${CMAKE_SYSTEM_NAME} PLATFORM_INDEX)
set(VULKAN_AVAILABLE ${VULKAN_UNSUPPORTED_PLATFORMS} LESS 0)

include(CMakeDependentOption)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(NN_IS_ROOT ON)
endif()

option(NN_SKIP_NEURALNET "Omit neuralnet build" OFF)
option(NN_BUILD_NETWORKS "Build example network programs" ${NN_IS_ROOT})
cmake_dependent_option(NN_BUILD_VULKAN "Build Vulkan headers & meta-loader. If NN_SUPPORT_VULKAN is enabled, and NN_BUILD_VULKAN is disabled, adding CMake targets for each library is required to build" ON "NOT NN_BUILD_NETWORKS" ON)

option(NN_SUPPORT_CPU "Support CPU evaluation & training" ON)
cmake_dependent_option(NN_SUPPORT_VULKAN "Support Vulkan compute shader evaluation & training" ON "VULKAN_AVAILABLE" OFF)

if(NN_SUPPORT_CPU)
    list(APPEND NN_EVALUATORS cpu)
endif()

if(NN_SUPPORT_VULKAN)
    list(APPEND NN_EVALUATORS vulkan)
endif()

add_subdirectory("vendor")
if(NOT ${NN_SKIP_NEURALNET})
    add_subdirectory("src")
endif()