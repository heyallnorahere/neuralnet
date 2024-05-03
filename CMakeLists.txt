cmake_minimum_required(VERSION 3.21.0)
project(neuralnet)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(VULKAN_UNSUPPORTED_PLATFORMS Darwin iOS)
list(FIND VULKAN_UNSUPPORTED_PLATFORMS ${CMAKE_SYSTEM_NAME} PLATFORM_INDEX)
set(VULKAN_AVAILABLE ${VULKAN_UNSUPPORTED_PLATFORMS} LESS 0)

include(CMakeDependentOption)

option(NN_SUPPORT_CPU "Support CPU evaluation & training" ON)
cmake_dependent_option(NN_SUPPORT_VULKAN "Support Vulkan compute shader evaluation & training" ON "VULKAN_AVAILABLE" OFF)

if(NN_SUPPORT_CPU)
    list(APPEND NN_EVALUATORS cpu)
endif()

if(NN_SUPPORT_VULKAN)
    list(APPEND NN_EVALUATORS vulkan)
endif()

add_subdirectory("vendor")
add_subdirectory("src")