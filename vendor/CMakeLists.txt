cmake_minimum_required(VERSION 3.21.0)

add_subdirectory("tracy")
add_subdirectory("zlib")
add_subdirectory("json")

if(NN_BUILD_VULKAN AND NN_SUPPORT_VULKAN)
    add_subdirectory("Vulkan-Headers")

    set(VOLK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/volk")
    add_library(volk STATIC "${VOLK_DIR}/volk.c" "${VOLK_DIR}/volk.h")
    target_link_libraries(volk PUBLIC Vulkan::Headers)
    target_include_directories(volk PUBLIC ${VOLK_DIR})

    set(VMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator")
    file(GLOB VMA_SOURCE CONFIGURE_DEPENDS "${VMA_DIR}/src/VmaUsage.*")
    list(APPEND VMA_SOURCE "${VMA_DIR}/include/vk_mem_alloc.h")

    if(WIN32 AND ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        list(APPEND VMA_SOURCE "${VMA_DIR}/src/vk_mem_alloc.natvis")
    endif()

    add_library(vma STATIC ${VMA_SOURCE})
    target_include_directories(vma PUBLIC "${VMA_DIR}/include")
    target_link_libraries(vma PUBLIC Vulkan::Headers)

    set_target_properties(vma PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

    target_compile_definitions(vma PRIVATE
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
        VMA_DEBUG_ALWAYS_DEDICATED_MEMORY=0
        VMA_DEBUG_INITIALIZE_ALLOCATIONS=0
        VMA_DEBUG_GLOBAL_MUTEX=0
        VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT=0
        VMA_RECORDING_ENABLED=0)

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/shaderc/utils/git-sync-deps")

    add_subdirectory("shaderc")
    if(NN_BUILD_NETWORKS)
        add_subdirectory("glfw")
    endif()
endif()