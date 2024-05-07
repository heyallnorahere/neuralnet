#include "nnpch.h"
#include "neuralnet/evaluators/evaluators.h"
#include "neuralnet/util.h"

namespace neuralnet::evaluators {
    static std::unique_ptr<vulkan_context_t> s_next_context;

    static VkBool32 vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                          VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                          const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                          void* pUserData) {
        ZoneScoped;

        switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            std::cerr << "Vulkan warning: " << pCallbackData->pMessage << std::endl;
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            throw std::runtime_error(pCallbackData->pMessage);
        default:
            std::cout << "Vulkan message: " << pCallbackData->pMessage << std::endl;
        }

        return VK_FALSE;
    }

    void vulkan_evaluator::set_next_context(std::unique_ptr<vulkan_context_t>&& context) {
        ZoneScoped;
        s_next_context = std::move(context);
    }

    bool vulkan_evaluator::is_context_valid() {
        ZoneScoped;
        if (!s_next_context) {
            return false;
        }

        if (s_next_context->vtable.vkGetInstanceProcAddr == nullptr) {
            return false;
        }

        if (s_next_context->handles.context_provided &&
            s_next_context->handles.vulkan_version == 0) {
            return false;
        }

        return true;
    }

    static std::unordered_map<void*, size_t> s_vulkan_block_sizes;

    static void* vk_alloc(void* pUserData, size_t size, size_t alignment,
                          VkSystemAllocationScope allocationScope) {
        ZoneScoped;

        void* ptr = alloc(size);
        s_vulkan_block_sizes.insert(std::make_pair(ptr, size));

        return ptr;
    }

    static void vk_free(void* pUserData, void* pMemory) {
        ZoneScoped;

        freemem(pMemory);
        s_vulkan_block_sizes.erase(pMemory);
    }

    static void* vk_realloc(void* pUserData, void* pOriginal, size_t size, size_t alignment,
                           VkSystemAllocationScope allocationScope) {
        ZoneScoped;

        size_t original_size = s_vulkan_block_sizes.at(pOriginal);
        void* new_ptr = vk_alloc(pUserData, size, alignment, allocationScope);

        std::memcpy(new_ptr, pOriginal, std::min(original_size, size));
        vk_free(pUserData, pOriginal);

        return new_ptr;
    }

#define NN_LOAD_VK_GLOBAL(vtable, name)                                                            \
    if (vtable.name == nullptr)                                                                    \
    vtable.name = (PFN_##name)vtable.vkGetInstanceProcAddr(nullptr, #name)

#define NN_LOAD_VK_INSTANCE(vtable, instance, name)                                                \
    if (vtable.name == nullptr)                                                                    \
    vtable.name = (PFN_##name)vtable.vkGetInstanceProcAddr(instance, #name)

#define NN_LOAD_VK_DEVICE(vtable, device, name)                                                    \
    if (vtable.name == nullptr)                                                                    \
    vtable.name = (PFN_##name)vtable.vkGetDeviceProcAddr(device, #name)

    static void vtable_load_globals(vulkan_vtable_t& vtable) {
        ZoneScoped;

        if (vtable.check_result == nullptr) {
            vtable.check_result = [](VkResult result) {
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("Non-success error code received!");
                }
            };
        }

        if (vtable.debug_callback == nullptr) {
            vtable.debug_callback = vulkan_debug_callback;
        }

        if (vtable.alloc_callbacks.pfnAllocation == nullptr) {
            vtable.alloc_callbacks.pfnAllocation = vk_alloc;
        }

        if (vtable.alloc_callbacks.pfnFree == nullptr) {
            vtable.alloc_callbacks.pfnFree = vk_free;
        }

        if (vtable.alloc_callbacks.pfnReallocation == nullptr) {
            vtable.alloc_callbacks.pfnReallocation = vk_realloc;
        }

        NN_LOAD_VK_GLOBAL(vtable, vkEnumerateInstanceExtensionProperties);
        NN_LOAD_VK_GLOBAL(vtable, vkEnumerateInstanceLayerProperties);
        NN_LOAD_VK_GLOBAL(vtable, vkCreateInstance);
    }

    static void vtable_load_instance(vulkan_vtable_t& vtable, VkInstance instance) {
        ZoneScoped;
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetDeviceProcAddr);

        NN_LOAD_VK_INSTANCE(vtable, instance, vkDestroyInstance);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkEnumeratePhysicalDevices);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetPhysicalDeviceFeatures);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetPhysicalDeviceFeatures2);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetPhysicalDeviceProperties);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetPhysicalDeviceQueueFamilyProperties);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkEnumerateDeviceExtensionProperties);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkCreateDevice);

        NN_LOAD_VK_INSTANCE(vtable, instance, vkCreateDebugUtilsMessengerEXT);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkDestroyDebugUtilsMessengerEXT);
    }

    static void vtable_load_device(vulkan_vtable_t& vtable, VkDevice device) {
        ZoneScoped;

        // creating objects
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateDescriptorSetLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateDescriptorPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkAllocateDescriptorSets);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreatePipelineLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateComputePipelines);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateCommandPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkAllocateCommandBuffers);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateFence);

        // destroying objects
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyFence);
        NN_LOAD_VK_DEVICE(vtable, device, vkFreeCommandBuffers);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyCommandPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyPipeline);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyPipelineLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkFreeDescriptorSets);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyDescriptorPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyDescriptorSetLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyDevice);

        // sync/queues
        NN_LOAD_VK_DEVICE(vtable, device, vkBeginCommandBuffer);
        NN_LOAD_VK_DEVICE(vtable, device, vkEndCommandBuffer);
        NN_LOAD_VK_DEVICE(vtable, device, vkGetDeviceQueue);
        NN_LOAD_VK_DEVICE(vtable, device, vkQueueSubmit);
        NN_LOAD_VK_DEVICE(vtable, device, vkGetFenceStatus);
        NN_LOAD_VK_DEVICE(vtable, device, vkWaitForFences);
        NN_LOAD_VK_DEVICE(vtable, device, vkResetFences);

        // commands
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdPipelineBarrier);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdBindPipeline);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdBindDescriptorSets);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdDispatch);
    }

    static void create_instance(vulkan_context_t* context) {
        ZoneScoped;
        const auto& v = context->vtable;

        static const std::unordered_set<std::string> requested_extensions = {
            "VK_KHR_get_physical_device_properties2",
#ifdef NN_DEBUG
            "VK_EXT_debug_utils",
#endif
        };

        static const std::unordered_set<std::string> requested_layers = {
#ifdef NN_DEBUG
            "VK_LAYER_KHRONOS_validation"
#endif
        };

        uint32_t extension_count, layer_count;
        v.check_result(
            v.vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr));
        v.check_result(v.vkEnumerateInstanceLayerProperties(&layer_count, nullptr));

        std::vector<VkExtensionProperties> extensions(extension_count);
        std::vector<VkLayerProperties> layers(layer_count);

        v.vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());
        v.vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

        std::vector<const char*> used_extensions, used_layers;
        for (const auto& extension : extensions) {
            if (requested_extensions.find(extension.extensionName) == requested_extensions.end()) {
                continue;
            }

            used_extensions.push_back(extension.extensionName);
        }

        for (const auto& layer : layers) {
            if (requested_layers.find(layer.layerName) == requested_layers.end()) {
                continue;
            }

            used_layers.push_back(layer.layerName);
        }

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.apiVersion = context->handles.vulkan_version;
        app_info.engineVersion = app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        app_info.pEngineName = app_info.pApplicationName = context->name.c_str();

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledExtensionCount = (uint32_t)used_extensions.size();
        create_info.ppEnabledExtensionNames = used_extensions.data();
        create_info.enabledLayerCount = (uint32_t)used_layers.size();
        create_info.ppEnabledLayerNames = used_layers.data();

        // todo: is this necessary?
        // create_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

        v.check_result(
            v.vkCreateInstance(&create_info, &v.alloc_callbacks, &context->handles.instance));
    }

    static void create_debug_messenger(vulkan_context_t* context) {
        ZoneScoped;

        const auto& v = context->vtable;
        if (v.vkCreateDebugUtilsMessengerEXT == nullptr) {
            context->handles.debug_messenger = nullptr;
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.pfnUserCallback = v.debug_callback;
        create_info.pUserData = context;

        create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;

        v.check_result(v.vkCreateDebugUtilsMessengerEXT(context->handles.instance, &create_info,
                                                        &v.alloc_callbacks,
                                                        &context->handles.debug_messenger));
    }

    static void get_device_families(VkPhysicalDevice device,
                                    std::vector<VkQueueFamilyProperties>& families,
                                    const vulkan_vtable_t& vtable) {
        ZoneScoped;

        uint32_t family_count = 0;
        vtable.vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);

        families.resize(family_count);
        vtable.vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
    }

    static constexpr VkQueueFlags vulkan_compute_flag =
        VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;

    static uint32_t score_device(VkPhysicalDevice device, uint32_t* compute_queue_index,
                                 vulkan_context_t* context) {
        ZoneScoped;
        const auto& v = context->vtable;

        VkPhysicalDeviceProperties properties;
        v.vkGetPhysicalDeviceProperties(device, &properties);

        // gotta make sure this device is viable
        std::vector<VkQueueFamilyProperties> families;
        get_device_families(device, families, v);

        bool found_compute = false;
        for (uint32_t i = 0; i < families.size(); i++) {

            const auto& family = families[i];
            if ((family.queueFlags & vulkan_compute_flag) == vulkan_compute_flag) {
                found_compute = true;
                *compute_queue_index = i;

                break;
            }
        }

        if (!found_compute) {
            std::string message =
                "Vulkan device \"" + std::string(properties.deviceName) + "\" not suitable";
            TracyMessageC(message.c_str(), message.length(), tracy::Color::Red);

            return 0;
        }

        // we want to maximize image size and compute work groups
        // we could use the cpu evaluator and a neural network but i dont Want to Do That

        uint32_t score = 0;
        score += properties.limits.maxImageDimension2D;

        for (size_t i = 0; i < 3; i++) {
            score += properties.limits.maxComputeWorkGroupCount[i];
        }

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 10000;
        }

        std::string score_message = "Vulkan device \"" + std::string(properties.deviceName) +
                                    "\" score: " + std::to_string(score);
        TracyMessageC(score_message.c_str(), score_message.length(), tracy::Color::Grey75);

        return score;
    }

    static void select_physical_device(vulkan_context_t* context) {
        ZoneScoped;
        const auto& v = context->vtable;

        uint32_t device_count;
        v.check_result(
            v.vkEnumeratePhysicalDevices(context->handles.instance, &device_count, nullptr));

        std::vector<VkPhysicalDevice> devices(device_count);
        v.vkEnumeratePhysicalDevices(context->handles.instance, &device_count, devices.data());

        uint32_t max_score = 0;
        size_t device_index = devices.size();
        uint32_t compute_queue_index;

        for (size_t i = 0; i < devices.size(); i++) {
            auto device = devices[i];

            uint32_t compute_index;
            uint32_t score = score_device(device, &compute_index, context);

            if (score > max_score) {
                max_score = score;
                device_index = i;
                compute_queue_index = compute_index;
            }
        }

        if (device_index >= devices.size()) {
            throw std::runtime_error("Failed to find a valid Vulkan device!");
        }

        context->handles.physical_device = devices[device_index];
        context->handles.compute_queue_index = compute_queue_index;
    }

    static void create_device(vulkan_context_t* context) {
        ZoneScoped;
        const auto& v = context->vtable;

        std::vector<VkQueueFamilyProperties> families;
        get_device_families(context->handles.physical_device, families, context->vtable);

        const auto& compute_family = families[context->handles.compute_queue_index];
        if ((compute_family.queueFlags & vulkan_compute_flag) != vulkan_compute_flag) {
            throw std::runtime_error("Queue flags are not valid!");
        }

        std::unordered_set<uint32_t> queue_family_indices(context->handles.shared_queue_indices);
        queue_family_indices.insert(context->handles.compute_queue_index);

        std::vector<VkDeviceQueueCreateInfo> queue_info;
        float priority = 1.f;

        for (uint32_t index : queue_family_indices) {
            auto& queue_create_info = queue_info.emplace_back();
            std::memset(&queue_create_info, 0, sizeof(VkDeviceQueueCreateInfo));

            queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_create_info.queueCount = 1;
            queue_create_info.pQueuePriorities = &priority;
            queue_create_info.queueFamilyIndex = index;
        }

        static const std::unordered_set<std::string> requested_extensions = {
            "VK_KHR_portability_subset"
        };

        uint32_t extension_count;
        v.check_result(v.vkEnumerateDeviceExtensionProperties(context->handles.physical_device,
                                                              nullptr, &extension_count, nullptr));

        std::vector<VkExtensionProperties> extensions(extension_count);
        v.vkEnumerateDeviceExtensionProperties(context->handles.physical_device, nullptr,
                                               &extension_count, extensions.data());

        std::vector<const char*> used_extensions;
        for (const auto& extension : extensions) {
            if (requested_extensions.find(extension.extensionName) == requested_extensions.end()) {
                continue;
            }

            used_extensions.push_back(extension.extensionName);
        }

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.enabledExtensionCount = (uint32_t)used_extensions.size();
        create_info.ppEnabledExtensionNames = used_extensions.data();
        create_info.queueCreateInfoCount = (uint32_t)queue_info.size();
        create_info.pQueueCreateInfos = queue_info.data();

        std::vector<void*> allocated_blocks;
        if (context->handles.vulkan_version < VK_API_VERSION_1_1) {
            auto features = (VkPhysicalDeviceFeatures*)alloc(sizeof(VkPhysicalDeviceFeatures));
            allocated_blocks.push_back(features);

            v.vkGetPhysicalDeviceFeatures(context->handles.physical_device, features);
            create_info.pEnabledFeatures = features;
        } else {
            auto features2 = (VkPhysicalDeviceFeatures2*)alloc(sizeof(VkPhysicalDeviceFeatures2));
            auto features11 =
                (VkPhysicalDeviceVulkan11Features*)alloc(sizeof(VkPhysicalDeviceVulkan11Features));

            allocated_blocks.push_back(features2);
            allocated_blocks.push_back(features11);

            features2->pNext = features11;
            features2->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features11->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

            if (context->handles.vulkan_version >= VK_API_VERSION_1_2) {
                auto features12 = (VkPhysicalDeviceVulkan12Features*)alloc(
                    sizeof(VkPhysicalDeviceVulkan12Features));
                allocated_blocks.push_back(features12);

                features11->pNext = features12;
                features12->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

                if (context->handles.vulkan_version >= VK_API_VERSION_1_3) {
                    auto features13 = (VkPhysicalDeviceVulkan13Features*)alloc(
                        sizeof(VkPhysicalDeviceVulkan13Features));
                    allocated_blocks.push_back(features13);

                    features12->pNext = features13;
                    features13->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
                }
            }

            v.vkGetPhysicalDeviceFeatures2(context->handles.physical_device, features2);
            create_info.pNext = features2;
        }

        v.check_result(v.vkCreateDevice(context->handles.physical_device, &create_info,
                                        &v.alloc_callbacks, &context->handles.device));

        for (void* block : allocated_blocks) {
            freemem(block);
        }

        VkPhysicalDeviceProperties properties;
        v.vkGetPhysicalDeviceProperties(context->handles.physical_device, &properties);

        std::string message = "Vulkan device created: " + std::string(properties.deviceName);
        TracyMessage(message.c_str(), message.length());
    }

    static void create_allocator(vulkan_context_t* context) {
        ZoneScoped;

        auto& v = context->vtable;
        auto& handles = context->handles;

        VmaVulkanFunctions vtable{};
        vtable.vkGetInstanceProcAddr = v.vkGetInstanceProcAddr;
        vtable.vkGetDeviceProcAddr = v.vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo create_info{};
        create_info.device = handles.device;
        create_info.physicalDevice = handles.physical_device;
        create_info.instance = handles.instance;
        create_info.pAllocationCallbacks = &context->vtable.alloc_callbacks;
        create_info.pDeviceMemoryCallbacks = nullptr;
        create_info.vulkanApiVersion = handles.vulkan_version;
        create_info.pHeapSizeLimit = nullptr;
        create_info.preferredLargeHeapBlockSize = 0;
        create_info.pVulkanFunctions = &vtable;
        create_info.pTypeExternalMemoryHandleTypes = nullptr;
        create_info.flags = 0; // none

        v.check_result(vmaCreateAllocator(&create_info, &handles.allocator));
    }

    static void create_objects(vulkan_evaluator_objects_t* objects, vulkan_context_t* context) {
        ZoneScoped;

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = handles.compute_queue_index;

        static constexpr uint32_t max_sets = 100;
        static const std::vector<VkDescriptorPoolSize> pool_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_sets * 5 }
        };

        VkDescriptorPoolCreateInfo descriptor_pool_info{};
        descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptor_pool_info.maxSets = max_sets;
        descriptor_pool_info.poolSizeCount = (uint32_t)pool_sizes.size();
        descriptor_pool_info.pPoolSizes = pool_sizes.data();

        v.vkGetDeviceQueue(handles.device, handles.compute_queue_index, 0, &objects->compute_queue);

        v.check_result(v.vkCreateCommandPool(handles.device, &command_pool_info, &v.alloc_callbacks,
                                             &objects->command_pool));

        v.check_result(v.vkCreateDescriptorPool(handles.device, &descriptor_pool_info,
                                                &v.alloc_callbacks, &objects->descriptor_pool));
    }

    static void create_profiler(vulkan_context_t* context, vulkan_evaluator_objects_t* objects) {
        ZoneScoped;

        auto& v = context->vtable;
        auto& handles = context->handles;

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = objects->command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer;
        v.vkAllocateCommandBuffers(handles.device, &alloc_info, &command_buffer);

        handles.profiler_context = TracyVkContext(
            handles.instance, handles.physical_device, handles.device, objects->compute_queue,
            command_buffer, v.vkGetInstanceProcAddr, v.vkGetDeviceProcAddr);

        v.vkFreeCommandBuffers(handles.device, objects->command_pool, 1, &command_buffer);
    }

    void vulkan_evaluator::init_vulkan() {
        ZoneScoped;

        if (m_context->name.empty()) {
            m_context->name = "neuralnet Vulkan context";
        }

        vtable_load_globals(m_context->vtable);
        if (!m_context->handles.context_provided) {
            if (m_context->handles.vulkan_version == 0) {
                // don't need any higher version than 1.0
                m_context->handles.vulkan_version = VK_API_VERSION_1_0;
            }

            create_instance(m_context.get());
        }

        vtable_load_instance(m_context->vtable, m_context->handles.instance);
        if (!m_context->handles.context_provided) {
            create_debug_messenger(m_context.get());

            select_physical_device(m_context.get());
            create_device(m_context.get());
        }

        vtable_load_device(m_context->vtable, m_context->handles.device);
        create_objects(&m_objects, m_context.get());

        if (m_context->handles.context_provided) {
            m_profiling_enabled = m_context->handles.profiler_context != nullptr;
        } else {
            m_profiling_enabled = true;
            create_profiler(m_context.get(), &m_objects);
        }

        // todo: further init
    }

    void vulkan_evaluator::shutdown_vulkan() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        v.vkDestroyCommandPool(handles.device, m_objects.command_pool, &v.alloc_callbacks);
        v.vkDestroyDescriptorPool(handles.device, m_objects.descriptor_pool, &v.alloc_callbacks);

        if (!m_context->handles.context_provided) {
            TracyVkDestroy(handles.profiler_context);

            vmaDestroyAllocator(handles.allocator);
            v.vkDestroyDevice(handles.device, &v.alloc_callbacks);

            if (handles.debug_messenger != nullptr) {
                v.vkDestroyDebugUtilsMessengerEXT(handles.instance, handles.debug_messenger,
                                                  &v.alloc_callbacks);
            }

            v.vkDestroyInstance(handles.instance, &v.alloc_callbacks);
        }
    }

    vulkan_evaluator::vulkan_evaluator() {
        ZoneScoped;
        if (!is_context_valid()) {
            throw std::runtime_error("No valid context!");
        }

        m_context = std::move(s_next_context);
        s_next_context.reset(); // just to be safe

        init_vulkan();
    }

    vulkan_evaluator::~vulkan_evaluator() {
        ZoneScoped;
        shutdown_vulkan();
    }

    bool vulkan_evaluator::is_result_ready(uint64_t result) {
        ZoneScoped;

        // todo: check
        return false;
    }

    bool vulkan_evaluator::free_result(uint64_t result) {
        ZoneScoped;

        // todo: check
        return false;
    }

    std::optional<uint64_t> vulkan_evaluator::begin_eval(const network* nn,
                                                         const std::vector<number_t>& inputs) {
        ZoneScoped;

        // todo: begin eval (put together VkImage?)
        return {};
    }

    std::optional<uint64_t> vulkan_evaluator::begin_eval(const network* nn, void* native_inputs) {
        ZoneScoped;

        // todo: begin eval (use input as VkImage?)
        return {};
    }

    bool vulkan_evaluator::get_eval_result(uint64_t result, void** outputs) {
        ZoneScoped;

        // todo: get eval result
        return false;
    }

    void vulkan_evaluator::retrieve_eval_values(const network* nn, void* native_outputs,
                                                std::vector<number_t>& outputs) {
        ZoneScoped;

        // todo: retrieve eval results (from VkImage?)
    }

    std::optional<uint64_t> vulkan_evaluator::begin_backprop(const network* nn,
                                                             const backprop_data_t& data) {
        ZoneScoped;

        // todo: backprop
        return {};
    }

    bool vulkan_evaluator::get_backprop_result(uint64_t result, std::vector<layer_t>& deltas) {
        ZoneScoped;

        // todo: get backprop result
        return false;
    }

    void vulkan_evaluator::flush() {
        ZoneScoped;

        // todo: wait for fences
    }

    number_t vulkan_evaluator::cost_function(number_t actual, number_t expected) {
        ZoneScoped;

        // todo: use actual cost function
        return 0;
    }
} // namespace neuralnet::evaluators