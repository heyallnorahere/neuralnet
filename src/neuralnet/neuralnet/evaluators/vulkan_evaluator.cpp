#include "nnpch.h"
#include "neuralnet/evaluators/evaluators.h"
#include "neuralnet/util.h"
#include "neuralnet/resources.h"

namespace neuralnet::evaluators {
    static std::unique_ptr<vulkan_context_t> s_next_context;

    // see resources/glsl/include/buffers.glsl
    static constexpr size_t max_layers = 32;

    static constexpr VkFormat image_format = VK_FORMAT_R32_SFLOAT;
    static constexpr VkImageTiling image_tiling = VK_IMAGE_TILING_OPTIMAL;
    static constexpr VkImageAspectFlags image_aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
    static constexpr VkImageUsageFlags image_usage = VK_IMAGE_USAGE_STORAGE_BIT |
                                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    static constexpr VkPipelineStageFlags compute_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    static constexpr VkImageLayout image_compute_layout = VK_IMAGE_LAYOUT_GENERAL;
    static constexpr VkAccessFlags image_access_flags =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    static constexpr VkAccessFlags transfer_src_access = VK_ACCESS_TRANSFER_READ_BIT;
    static constexpr VkAccessFlags transfer_dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
    static constexpr VkImageLayout transfer_src_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    static constexpr VkImageLayout transfer_dst_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    static constexpr VkPipelineStageFlags transfer_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    static VkBool32 vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                          VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                          const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                          void* pUserData) {
        ZoneScoped;

        std::string severity;
        switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            severity = "warning";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            severity = "error";
            break;
        default:
            std::cout << "Vulkan message: " << pCallbackData->pMessage << std::endl;
        }

        std::cerr << "neuralnet Vulkan " << severity << ": " << pCallbackData->pMessage
                  << std::endl;
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

    static void* vk_alloc(void* pUserData, size_t size, size_t alignment,
                          VkSystemAllocationScope allocationScope) {
        ZoneScoped;
        return alloc(size);
    }

    static void vk_free(void* pUserData, void* pMemory) {
        ZoneScoped;
        freemem(pMemory);
    }

    static void* vk_realloc(void* pUserData, void* pOriginal, size_t size, size_t alignment,
                            VkSystemAllocationScope allocationScope) {
        ZoneScoped;
        return reallocate(pOriginal, size);
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
                    throw std::runtime_error("Non-success Vulkan result received!");
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
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetPhysicalDeviceImageFormatProperties);
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
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateShaderModule);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreatePipelineLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateComputePipelines);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateCommandPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkAllocateCommandBuffers);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateFence);
        NN_LOAD_VK_DEVICE(vtable, device, vkCreateImageView);

        // destroying objects
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyImageView);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyFence);
        NN_LOAD_VK_DEVICE(vtable, device, vkFreeCommandBuffers);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyCommandPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyPipeline);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyPipelineLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyShaderModule);
        NN_LOAD_VK_DEVICE(vtable, device, vkFreeDescriptorSets);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyDescriptorPool);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyDescriptorSetLayout);
        NN_LOAD_VK_DEVICE(vtable, device, vkDestroyDevice);

        // sync/queues
        NN_LOAD_VK_DEVICE(vtable, device, vkBeginCommandBuffer);
        NN_LOAD_VK_DEVICE(vtable, device, vkEndCommandBuffer);
        NN_LOAD_VK_DEVICE(vtable, device, vkGetDeviceQueue);
        NN_LOAD_VK_DEVICE(vtable, device, vkQueueSubmit);
        NN_LOAD_VK_DEVICE(vtable, device, vkQueueWaitIdle);
        NN_LOAD_VK_DEVICE(vtable, device, vkGetFenceStatus);
        NN_LOAD_VK_DEVICE(vtable, device, vkWaitForFences);
        NN_LOAD_VK_DEVICE(vtable, device, vkResetFences);

        // commands
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdPipelineBarrier);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdBindPipeline);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdBindDescriptorSets);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdPushConstants);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdDispatch);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdCopyBufferToImage);
        NN_LOAD_VK_DEVICE(vtable, device, vkCmdCopyImageToBuffer);

        // idk man
        NN_LOAD_VK_DEVICE(vtable, device, vkUpdateDescriptorSets);
    }

    static void create_vulkan_buffer(vulkan_context_t* context, size_t size,
                                     vulkan_buffer_t* buffer) {
        ZoneScoped;
        buffer->size = size;

        VkBufferCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = (VkDeviceSize)size;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // not for any other purpose
        create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

        context->vtable.check_result(vmaCreateBuffer(context->handles.allocator, &create_info,
                                                     &alloc_info, &buffer->buffer,
                                                     &buffer->allocation, nullptr));
    }

    static void destroy_vulkan_buffer(vulkan_context_t* context, const vulkan_buffer_t* buffer) {
        ZoneScoped;
        vmaDestroyBuffer(context->handles.allocator, buffer->buffer, buffer->allocation);
    }

    static void create_vulkan_image(vulkan_context_t* context, VkImageType type,
                                    VkImageViewType view_type, const VkExtent3D& size,
                                    vulkan_image_t* image) {
        ZoneScoped;

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        image->size = size;
        image->type = type;
        image->view_type = view_type;

        std::vector<uint32_t> queue_indices = { context->handles.compute_queue_index };
        for (uint32_t index : context->handles.shared_queue_indices) {
            if (index != context->handles.compute_queue_index) {
                queue_indices.push_back(index);
            }
        }

        VkImageCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create_info.imageType = type;
        create_info.extent = size;
        create_info.arrayLayers = 1;
        create_info.mipLevels = 1;
        create_info.format = image_format;
        create_info.tiling = image_tiling;
        create_info.usage = image_usage | context->handles.additional_image_usage;
        create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        create_info.samples = VK_SAMPLE_COUNT_1_BIT;

        if (queue_indices.size() > 1) {
            create_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = (uint32_t)queue_indices.size();
            create_info.pQueueFamilyIndices = queue_indices.data();
        } else {
            create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        v.check_result(vmaCreateImage(handles.allocator, &create_info, &alloc_info, &image->image,
                                      &image->allocation, nullptr));

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image->image;
        view_info.viewType = view_type;
        view_info.format = image_format;
        view_info.subresourceRange.aspectMask = image_aspect_flags;
        view_info.subresourceRange.layerCount = 1;
        view_info.subresourceRange.levelCount = 1;

        v.check_result(
            v.vkCreateImageView(handles.device, &view_info, &v.alloc_callbacks, &image->view));
    }

    static void destroy_vulkan_image(vulkan_context_t* context, const vulkan_image_t* image) {
        ZoneScoped;

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        v.vkDestroyImageView(handles.device, image->view, &v.alloc_callbacks);
        vmaDestroyImage(handles.allocator, image->image, image->allocation);
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
            if (!requested_extensions.contains(extension.extensionName) &&
                !context->handles.instance_extensions.contains(extension.extensionName)) {
                continue;
            }

            used_extensions.push_back(extension.extensionName);
        }

        for (const auto& layer : layers) {
            if (!requested_layers.contains(layer.layerName)) {
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
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

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

    static bool device_supports_format(VkPhysicalDevice device, VkImageTiling tiling,
                                       VkFormat format, vulkan_context_t* context) {
        ZoneScoped;

        static const std::unordered_set<VkImageType> image_types = { VK_IMAGE_TYPE_2D,
                                                                     VK_IMAGE_TYPE_3D };

        const auto& v = context->vtable;
        for (auto type : image_types) {
            VkImageFormatProperties properties{};
            VkResult result = v.vkGetPhysicalDeviceImageFormatProperties(
                device, format, type, tiling, image_usage | context->handles.additional_image_usage,
                0, &properties);

            if (result != VK_SUCCESS) {
                return false;
            }
        }

        return true;
    }

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

        if (!found_compute ||
            !device_supports_format(device, image_tiling, image_format, context)) {
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

        VkPhysicalDeviceProperties properties;
        v.vkGetPhysicalDeviceProperties(context->handles.physical_device, &properties);

        std::cout << "neuralnet: chose compute device: " << properties.deviceName << std::endl;
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
            if (!requested_extensions.contains(extension.extensionName) &&
                !context->handles.device_extensions.contains(extension.extensionName)) {
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

    static void create_set_layout(vulkan_context_t* context, VkDescriptorSetLayout* layout,
                                  const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        ZoneScoped;

        VkDescriptorSetLayoutCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = (uint32_t)bindings.size();
        create_info.pBindings = bindings.data();

        const auto& v = context->vtable;
        v.check_result(v.vkCreateDescriptorSetLayout(context->handles.device, &create_info,
                                                     &v.alloc_callbacks, layout));
    }

    static void create_objects(vulkan_evaluator_objects_t* objects, vulkan_context_t* context) {
        ZoneScoped;

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = handles.compute_queue_index;

        static constexpr uint32_t max_sets = 200;
        static const std::vector<VkDescriptorPoolSize> pool_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, max_sets * 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_sets }
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

        static const std::vector<VkDescriptorSetLayoutBinding> evaluation_bindings = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT }
        };

        static const std::vector<VkDescriptorSetLayoutBinding> network_bindings = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT }
        };

        static const std::vector<std::string> shader_names = { "evaluation", "backpropagation",
                                                               "deltas" };

        create_set_layout(context, &objects->evaluation_layout, evaluation_bindings);
        create_set_layout(context, &objects->network_layout, network_bindings);

        std::vector<VkDescriptorSetLayout> set_layouts = { objects->evaluation_layout,
                                                           objects->network_layout };

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.size = sizeof(uint32_t) + sizeof(float); // see include/buffers.glsl
        range.offset = 0;

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = (uint32_t)set_layouts.size();
        layout_info.pSetLayouts = set_layouts.data();
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;

        v.check_result(v.vkCreatePipelineLayout(handles.device, &layout_info, &v.alloc_callbacks,
                                                &objects->pipeline_layout));

        std::vector<VkShaderModule> modules;
        std::vector<VkComputePipelineCreateInfo> pipeline_specs;

        for (const auto& name : shader_names) {
            std::string shader_path = "neuralnet/resources/spirv/" + name + ".spv";
            const auto& shader_resource = resource::get(shader_path);

            VkShaderModuleCreateInfo module_info{};
            module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            module_info.codeSize = shader_resource.size();
            module_info.pCode = (const uint32_t*)(const void*)shader_resource.data();

            VkShaderModule module;
            v.check_result(
                v.vkCreateShaderModule(handles.device, &module_info, &v.alloc_callbacks, &module));

            VkComputePipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline_info.layout = objects->pipeline_layout;
            pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipeline_info.stage.module = module;
            pipeline_info.stage.pName = "main";
            pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

            modules.push_back(module);
            pipeline_specs.push_back(pipeline_info);
        }

        std::vector<VkPipeline> pipelines(pipeline_specs.size());
        v.check_result(v.vkCreateComputePipelines(
            handles.device, VK_NULL_HANDLE, (uint32_t)pipeline_specs.size(), pipeline_specs.data(),
            &v.alloc_callbacks, pipelines.data()));

        for (size_t i = 0; i < pipelines.size(); i++) {
            v.vkDestroyShaderModule(handles.device, modules[i], &v.alloc_callbacks);

            objects->pipelines[shader_names[i]] = pipelines[i];
        }
    }

#ifdef TRACY_ENABLE
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
#endif

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

            user_callback_t chosen = m_context->vtable.user_callbacks.device_chosen;
            if (chosen != nullptr) {
                chosen(m_context.get());
            }

            create_device(m_context.get());
        }

        vtable_load_device(m_context->vtable, m_context->handles.device);
        create_objects(&m_objects, m_context.get());

        if (m_context->handles.context_provided) {
            m_profiling_enabled = m_context->handles.profiler_context != nullptr;
        } else {
#ifdef TRACY_ENABLE
            m_profiling_enabled = true;
            create_profiler(m_context.get(), &m_objects);
#else
            m_profiling_enabled = false;
#endif

            create_allocator(m_context.get());
        }

        user_callback_t init_finished = m_context->vtable.user_callbacks.init_finished;
        if (init_finished != nullptr) {
            init_finished(m_context.get());
        }
    }

    void vulkan_evaluator::shutdown_vulkan() {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        for (const auto& [name, pipeline] : m_objects.pipelines) {
            v.vkDestroyPipeline(handles.device, pipeline, &v.alloc_callbacks);
        }

        v.vkDestroyPipelineLayout(handles.device, m_objects.pipeline_layout, &v.alloc_callbacks);
        v.vkDestroyDescriptorSetLayout(handles.device, m_objects.evaluation_layout,
                                       &v.alloc_callbacks);
        v.vkDestroyDescriptorSetLayout(handles.device, m_objects.network_layout,
                                       &v.alloc_callbacks);

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

        m_current_pass_id = 0;
        m_current_result_id = 0;
    }

    vulkan_evaluator::~vulkan_evaluator() {
        ZoneScoped;
        set_training(false);

        std::vector<uint64_t> results;
        for (auto& [id, data] : m_results) {
            results.push_back(id);
        }

        for (uint64_t id : results) {
            free_result(id);
        }

        std::vector<const network*> networks;
        for (auto& [nn, data] : m_network_data) {
            data.references = 1;
            networks.push_back(nn);
        }

        for (auto nn : networks) {
            remove_network_reference(nn);
        }

        shutdown_vulkan();
    }

    bool vulkan_evaluator::is_result_ready(uint64_t result) const {
        ZoneScoped;

        if (!m_results.contains(result)) {
            return false;
        }

        const auto& result_data = m_results.at(result);
        VkResult status =
            m_context->vtable.vkGetFenceStatus(m_context->handles.device, result_data.fence);

        return status == VK_SUCCESS;
    }

    bool vulkan_evaluator::free_result(uint64_t result) {
        ZoneScoped;

        if (!is_result_ready(result)) {
            return false;
        }

        const auto& result_data = m_results.at(result);
        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        v.vkDestroyFence(handles.device, result_data.fence, &v.alloc_callbacks);
        v.vkFreeCommandBuffers(handles.device, m_objects.command_pool, 1,
                               &result_data.command_buffer);

        for (const auto& buffer : result_data.staging_buffers) {
            destroy_vulkan_buffer(m_context.get(), &buffer);
        }

        uint64_t pass = result_data.pass;
        m_results.erase(result);

        remove_pass_reference(pass);
        return true;
    }

    static VkCommandBuffer alloc_open_command_buffer(vulkan_context_t* context,
                                                     VkCommandPool command_pool) {
        ZoneScoped;

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandBufferCount = 1;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VkCommandBuffer command_buffer;
        v.check_result(v.vkAllocateCommandBuffers(handles.device, &alloc_info, &command_buffer));
        v.check_result(v.vkBeginCommandBuffer(command_buffer, &begin_info));

        return command_buffer;
    }

    static void end_and_submit_command_buffer(vulkan_context_t* context, VkQueue queue,
                                              VkCommandBuffer command_buffer, bool wait,
                                              VkFence fence) {
        ZoneScoped;
        const auto& v = context->vtable;

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        VkFence used_fence = fence;
        if (wait && used_fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

            v.check_result(v.vkCreateFence(context->handles.device, &create_info,
                                           &v.alloc_callbacks, &used_fence));
        }

        v.check_result(v.vkEndCommandBuffer(command_buffer));
        v.check_result(v.vkQueueSubmit(queue, 1, &submit_info, used_fence));

        if (wait) {
            v.check_result(v.vkWaitForFences(context->handles.device, 1, &used_fence, VK_TRUE,
                                             std::numeric_limits<uint64_t>::max()));

            if (fence == VK_NULL_HANDLE) {
                v.vkDestroyFence(context->handles.device, used_fence, &v.alloc_callbacks);
            }
        }
    }

    static void new_vulkan_result(vulkan_context_t* context, VkCommandPool pool,
                                  vulkan_result_t* result) {
        ZoneScoped;

        result->command_buffer = alloc_open_command_buffer(context, pool);
        if (!context->handles.context_provided) {
            TracyVkCollect(context->handles.profiler_context, result->command_buffer);
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        const auto& v = context->vtable;
        v.check_result(v.vkCreateFence(context->handles.device, &fence_info, &v.alloc_callbacks,
                                       &result->fence));
    }

    std::optional<uint64_t> vulkan_evaluator::begin_eval(const network* nn,
                                                         const std::vector<number_t>& inputs) {
        ZoneScoped;

        // i give up. just pass a vector pointer
        return begin_eval(nn, (void*)&inputs);
    }

    static constexpr uint32_t kernel_size = 8;
    static uint32_t get_work_group_count(uint64_t required_cells) {
        uint64_t remainder = required_cells % kernel_size;
        return (uint32_t)((required_cells - remainder) / kernel_size) + 1;
    }

    static void create_image_barrier(VkImageMemoryBarrier& barrier, VkImage image,
                                     VkAccessFlags src_access, VkAccessFlags dst_access,
                                     VkImageLayout src_layout, VkImageLayout dst_layout) {
        ZoneScoped;
        std::memset(&barrier, 0, sizeof(VkImageMemoryBarrier));

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcAccessMask = src_access;
        barrier.dstAccessMask = dst_access;
        barrier.oldLayout = src_layout;
        barrier.newLayout = dst_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = image_aspect_flags;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
    }

    std::optional<uint64_t> vulkan_evaluator::begin_eval(const network* nn, void* native_inputs) {
        ZoneScoped;
        const auto& inputs = *(const std::vector<number_t>*)native_inputs;

        uint64_t pass = new_pass(nn, inputs);
        uint64_t result = m_current_result_id++;

        auto& result_data = m_results[result];
        new_vulkan_result(m_context.get(), m_objects.command_pool, &result_data);

        result_data.pass = pass;
        result_data.type = vulkan_result_type::eval;

        const auto& network_data = m_network_data.at(nn);
        const auto& pass_data = m_passes.at(pass);
        const auto& v = m_context->vtable;

        VkPipeline pipeline = m_objects.pipelines.at("evaluation");
        v.vkCmdBindPipeline(result_data.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        std::vector<VkDescriptorSet> descriptor_sets = { pass_data.descriptor_set,
                                                         network_data.descriptor_set };

        v.vkCmdBindDescriptorSets(result_data.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_objects.pipeline_layout, 0, 2, descriptor_sets.data(), 0,
                                  nullptr);

        std::vector<VkImageMemoryBarrier> image_barriers;
        std::vector<VkImage> protected_images = { pass_data.activations.image, pass_data.z.image };

        for (VkImage image : protected_images) {
            auto& barrier = image_barriers.emplace_back();
            create_image_barrier(barrier, image, image_access_flags, image_access_flags,
                                 image_compute_layout, image_compute_layout);
        }

        {
            TracyVkZoneTransient(m_context->handles.profiler_context, vk_zone,
                                 result_data.command_buffer, "Network evaluation",
                                 m_profiling_enabled);

            const auto& layers = nn->get_layers();
            for (uint32_t i = 0; i < layers.size(); i++) {
                if (i > 0) {
                    static constexpr VkPipelineStageFlags stage =
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                    v.vkCmdPipelineBarrier(result_data.command_buffer, stage, stage, 0, 0, nullptr,
                                           0, nullptr, (uint32_t)image_barriers.size(),
                                           image_barriers.data());
                }

                v.vkCmdPushConstants(result_data.command_buffer, m_objects.pipeline_layout,
                                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &i);

                VkExtent3D work_groups;
                work_groups.width = get_work_group_count(layers[i].size);
                work_groups.height = get_work_group_count(pass_data.run_count);
                work_groups.depth = 1;

                v.vkCmdDispatch(result_data.command_buffer, work_groups.width, work_groups.height,
                                work_groups.depth);
            }
        }

        end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue,
                                      result_data.command_buffer, false, result_data.fence);

        return result;
    }

    bool vulkan_evaluator::get_eval_result(uint64_t result, void** outputs) {
        ZoneScoped;
        if (!m_results.contains(result)) {
            return false;
        }

        *outputs = get_pass_data(result);
        return true;
    }

    void vulkan_evaluator::retrieve_eval_values(const network* nn, void* native_outputs,
                                                std::vector<number_t>& outputs) {
        ZoneScoped;

        auto pass = (vulkan_pass_data_t*)native_outputs;
        auto& activations = pass->activations;

        VkImageMemoryBarrier src_barrier, dst_barrier;
        create_image_barrier(src_barrier, activations.image, image_access_flags,
                             transfer_src_access, image_compute_layout, transfer_src_layout);
        create_image_barrier(dst_barrier, activations.image, transfer_src_access,
                             image_access_flags, transfer_src_layout, image_compute_layout);

        const auto& layers = nn->get_layers();
        size_t layer_count = layers.size();
        const auto& last_layer = layers[layer_count - 1];

        VkBufferImageCopy image_copy{};
        image_copy.bufferOffset = 0;
        image_copy.imageOffset.y = (int32_t)activations.size.height - 2;
        image_copy.imageExtent.width = (uint32_t)last_layer.size;
        image_copy.imageExtent.height = 1;
        image_copy.imageExtent.depth = (uint32_t)pass->run_count;
        image_copy.imageSubresource.aspectMask = image_aspect_flags;
        image_copy.imageSubresource.baseArrayLayer = 0;
        image_copy.imageSubresource.layerCount = 1;
        image_copy.imageSubresource.mipLevel = 0;

        outputs.resize((size_t)last_layer.size * pass->run_count);

        vulkan_buffer_t staging_buffer;
        create_vulkan_buffer(m_context.get(), outputs.size() * sizeof(number_t), &staging_buffer);

        auto command_buffer = alloc_open_command_buffer(m_context.get(), m_objects.command_pool);
        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        {
            TracyVkZoneTransient(m_context->handles.profiler_context, transfer_zone, command_buffer,
                                 "Transfer eval outputs", m_profiling_enabled);

            v.vkCmdPipelineBarrier(command_buffer, compute_stage, transfer_stage, 0, 0, nullptr, 0,
                                   nullptr, 1, &src_barrier);

            v.vkCmdCopyImageToBuffer(command_buffer, activations.image, transfer_src_layout,
                                     staging_buffer.buffer, 1, &image_copy);

            v.vkCmdPipelineBarrier(command_buffer, transfer_stage, compute_stage, 0, 0, nullptr, 0,
                                   nullptr, 1, &dst_barrier);
        }

        end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue, command_buffer,
                                      true, VK_NULL_HANDLE);

        void* mapped = nullptr;
        v.check_result(vmaMapMemory(handles.allocator, staging_buffer.allocation, &mapped));

        copy(mapped, outputs.data(), staging_buffer.size);

        vmaUnmapMemory(handles.allocator, staging_buffer.allocation);
        destroy_vulkan_buffer(m_context.get(), &staging_buffer);
        v.vkFreeCommandBuffers(handles.device, m_objects.command_pool, 1, &command_buffer);
    }

    std::optional<uint64_t> vulkan_evaluator::begin_backprop(const network* nn,
                                                             const backprop_data_t& data) {
        ZoneScoped;

        auto& pass_data = *(vulkan_pass_data_t*)data.eval_outputs;
        uint64_t pass = pass_data.pass_id;
        uint64_t result = m_current_result_id++;

        auto& result_data = m_results[result];
        new_vulkan_result(m_context.get(), m_objects.command_pool, &result_data);

        result_data.pass = pass;
        result_data.type = vulkan_result_type::backprop;

        const auto& network_data = m_network_data.at(nn);
        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        VkPipeline pipeline = m_objects.pipelines.at("backpropagation");
        v.vkCmdBindPipeline(result_data.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        std::vector<VkDescriptorSet> descriptor_sets = { pass_data.descriptor_set,
                                                         network_data.descriptor_set };

        v.vkCmdBindDescriptorSets(result_data.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_objects.pipeline_layout, 0, 2, descriptor_sets.data(), 0,
                                  nullptr);

        std::vector<VkImageMemoryBarrier> image_barriers;
        std::vector<VkImage> protected_images = { pass_data.activations.image, pass_data.z.image };

        for (VkImage image : protected_images) {
            auto& barrier = image_barriers.emplace_back();
            create_image_barrier(barrier, image, image_access_flags, image_access_flags,
                                 image_compute_layout, image_compute_layout);
        }

        VkImageMemoryBarrier src_barrier, dst_barrier;
        create_image_barrier(src_barrier, pass_data.activations.image, image_access_flags,
                             transfer_dst_access, image_compute_layout, transfer_dst_layout);
        create_image_barrier(dst_barrier, pass_data.activations.image, transfer_dst_access,
                             image_access_flags, transfer_dst_layout, image_compute_layout);

        auto& staging_buffer = result_data.staging_buffers.emplace_back();
        create_vulkan_buffer(m_context.get(), data.expected_outputs.size() * sizeof(number_t),
                             &staging_buffer);

        void* mapped = nullptr;
        v.check_result(vmaMapMemory(handles.allocator, staging_buffer.allocation, &mapped));
        copy(data.expected_outputs.data(), mapped, staging_buffer.size);
        vmaUnmapMemory(handles.allocator, staging_buffer.allocation);

        const auto& layers = nn->get_layers();
        uint64_t output_count = layers[layers.size() - 1].size;

        std::vector<VkBufferImageCopy> regions;
        for (size_t i = 0; i < pass_data.run_count; i++) {
            auto& region = regions.emplace_back();
            std::memset(&region, 0, sizeof(VkBufferImageCopy));

            region.imageExtent.width = (uint32_t)output_count;
            region.imageExtent.height = 1;
            region.imageExtent.depth = 1;
            region.imageOffset.y = (uint32_t)pass_data.activations.size.height - 1;
            region.imageOffset.z = (uint32_t)i;
            region.imageSubresource.aspectMask = image_aspect_flags;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageSubresource.mipLevel = 0;
        }

        {
            TracyVkZoneTransient(m_context->handles.profiler_context, vk_zone,
                                 result_data.command_buffer, "Network backpropagation",
                                 m_profiling_enabled);

            v.vkCmdPipelineBarrier(result_data.command_buffer, compute_stage, transfer_stage, 0, 0,
                                   nullptr, 0, nullptr, 1, &src_barrier);

            v.vkCmdCopyBufferToImage(result_data.command_buffer, staging_buffer.buffer,
                                     pass_data.activations.image, transfer_dst_layout,
                                     (uint32_t)regions.size(), regions.data());

            v.vkCmdPipelineBarrier(result_data.command_buffer, transfer_stage, compute_stage, 0, 0,
                                   nullptr, 0, nullptr, 1, &dst_barrier);

            const auto& layers = nn->get_layers();
            for (uint32_t i = 0; i < layers.size(); i++) {
                if (i > 0) {
                    static constexpr VkPipelineStageFlags stage =
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                    v.vkCmdPipelineBarrier(result_data.command_buffer, stage, stage, 0, 0, nullptr,
                                           0, nullptr, (uint32_t)image_barriers.size(),
                                           image_barriers.data());
                }

                uint32_t layer_index = (uint32_t)layers.size() - (i + 1);
                v.vkCmdPushConstants(result_data.command_buffer, m_objects.pipeline_layout,
                                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
                                     &layer_index);

                VkExtent3D work_groups;
                work_groups.width = get_work_group_count(layers[i].size);
                work_groups.height = get_work_group_count(pass_data.run_count);
                work_groups.depth = 1;

                v.vkCmdDispatch(result_data.command_buffer, work_groups.width, work_groups.height,
                                work_groups.depth);
            }
        }

        end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue,
                                      result_data.command_buffer, false, result_data.fence);

        pass_data.references++;
        return result;
    }

    void vulkan_evaluator::copy_network_from_gpu(network* nn) {
        ZoneScoped;

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;
        const auto& network_data = m_network_data.at(nn);

        vulkan_buffer_t staging_buffer;
        std::vector<VkBufferImageCopy> regions;
        size_t data_size = 0;

        auto& layers = nn->get_layers();
        for (size_t i = 0; i < nn->get_layers().size(); i++) {
            const auto& layer = layers[i];

            VkBufferImageCopy region{};
            region.bufferOffset = (VkDeviceSize)data_size;
            region.imageOffset.z = (uint32_t)i;
            region.imageExtent.width = (uint32_t)layer.previous_size + 1;
            region.imageExtent.height = (uint32_t)layer.size;
            region.imageExtent.depth = 1;
            region.imageSubresource.aspectMask = image_aspect_flags;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageSubresource.mipLevel = 0;
            regions.push_back(region);

            data_size += sizeof(number_t) * region.imageExtent.width * region.imageExtent.height;
        }

        VkCommandBuffer command_buffer =
            alloc_open_command_buffer(m_context.get(), m_objects.command_pool);
        create_vulkan_buffer(m_context.get(), data_size, &staging_buffer);

        {
            TracyVkZoneTransient(handles.profiler_context, vk_zone, command_buffer,
                                 "Copy network from GPU", m_profiling_enabled);

            VkImageMemoryBarrier src_barrier, dst_barrier;
            create_image_barrier(src_barrier, network_data.data_image.image, image_access_flags,
                                 transfer_src_access, image_compute_layout, transfer_src_layout);
            create_image_barrier(dst_barrier, network_data.data_image.image, transfer_src_access,
                                 image_access_flags, transfer_src_layout, image_compute_layout);

            v.vkCmdPipelineBarrier(command_buffer, compute_stage, transfer_stage, 0, 0, nullptr, 0,
                                   nullptr, 1, &src_barrier);

            v.vkCmdCopyImageToBuffer(command_buffer, network_data.data_image.image,
                                     transfer_src_layout, staging_buffer.buffer,
                                     (uint32_t)regions.size(), regions.data());

            v.vkCmdPipelineBarrier(command_buffer, transfer_stage, compute_stage, 0, 0, nullptr, 0,
                                   nullptr, 1, &dst_barrier);
        }

        end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue, command_buffer,
                                      true, VK_NULL_HANDLE);

        v.vkFreeCommandBuffers(handles.device, m_objects.command_pool, 1, &command_buffer);

        number_t* mapped = nullptr;
        v.check_result(vmaMapMemory(handles.allocator, staging_buffer.allocation, (void**)&mapped));

        size_t offset = 0;
        for (auto& layer : layers) {
            for (uint64_t c = 0; c < layer.size; c++) {
                size_t current_offset = offset + c * (layer.previous_size + 1);
                layer.biases[c] = mapped[current_offset];

                copy(&mapped[current_offset + 1], &layer.weights[c * layer.previous_size],
                     layer.previous_size * sizeof(number_t));
            }

            offset += layer.size * (layer.previous_size + 1);
        }

        vmaUnmapMemory(handles.allocator, staging_buffer.allocation);
        destroy_vulkan_buffer(m_context.get(), &staging_buffer);
    }

    bool vulkan_evaluator::compose_deltas(const delta_composition_data_t& data) {
        ZoneScoped;
        if (!m_network_data.contains(data.nn)) {
            return false;
        }

        for (uint64_t key : data.backprop_keys) {
            if (!is_result_ready(key)) {
                return false;
            }
        }

        auto& layers = data.nn->get_layers();
        auto command_buffer = alloc_open_command_buffer(m_context.get(), m_objects.command_pool);

        const auto& handles = m_context->handles;
        const auto& v = m_context->vtable;
        const auto& network_data = m_network_data.at(data.nn);

        std::vector<VkDescriptorSet> sets(2, VK_NULL_HANDLE);
        sets[1] = network_data.descriptor_set;

        v.vkCmdPushConstants(command_buffer, m_objects.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             sizeof(uint32_t), sizeof(float), &data.delta_scalar);

        VkPipeline pipeline = m_objects.pipelines.at("deltas");
        v.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        VkImageMemoryBarrier sync_barrier;
        create_image_barrier(sync_barrier, network_data.data_image.image, image_access_flags,
                             image_access_flags, image_compute_layout, image_compute_layout);

        for (size_t i = 0; i < data.backprop_keys.size(); i++) {
            TracyVkZoneTransient(handles.profiler_context, vk_zone, command_buffer,
                                 "Compose deltas", m_profiling_enabled);

            uint64_t key = data.backprop_keys[i];
            const auto& result = m_results.at(key);
            const auto& pass = m_passes.at(result.pass);

            if (pass.nn != data.nn) {
                throw std::runtime_error("Network mismatch!");
            }

            if (i > 0) {
                v.vkCmdPipelineBarrier(command_buffer, compute_stage, compute_stage, 0, 0, nullptr,
                                       0, nullptr, 1, &sync_barrier);
            }

            sets[0] = pass.descriptor_set;
            v.vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      m_objects.pipeline_layout, 0, (uint32_t)sets.size(),
                                      sets.data(), 0, nullptr);

            VkExtent3D data_size = network_data.data_image.size;
            uint32_t x = get_work_group_count(data_size.width);
            uint32_t y = get_work_group_count(data_size.height);
            uint32_t z = get_work_group_count(data_size.depth);

            v.vkCmdDispatch(command_buffer, x, y, z);
        }

        end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue, command_buffer,
                                      true, VK_NULL_HANDLE);

        v.vkFreeCommandBuffers(handles.device, m_objects.command_pool, 1, &command_buffer);

        if (data.copy) {
            copy_network_from_gpu(data.nn);
        }

        return true;
    }

    number_t vulkan_evaluator::cost_function(number_t actual, number_t expected) const {
        ZoneScoped;

        // (x - y)^2
        // see include/functions.glsl

        number_t square_root = actual - expected;
        return square_root * square_root;
    }

    vulkan_context_t* vulkan_evaluator::get_context() { return m_context.get(); }

    vulkan_network_data_t* vulkan_evaluator::get_network_data(const network* network) {
        ZoneScoped;

        if (!m_network_data.contains(network)) {
            return nullptr;
        }

        return &m_network_data[network];
    }

    vulkan_pass_data_t* vulkan_evaluator::get_pass_data(uint64_t result) {
        ZoneScoped;

        const auto& result_data = m_results.at(result);
        return &m_passes[result_data.pass];
    }

    struct vulkan_layer_t {
        uint32_t size, previous_size, activation_function;
    };

    static void alloc_descriptor_sets(vulkan_context_t* context, VkDescriptorSetLayout layout,
                                      VkDescriptorPool pool, size_t set_count,
                                      VkDescriptorSet* sets) {
        ZoneScoped;
        std::vector<VkDescriptorSetLayout> layouts(set_count, layout);

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorSetCount = (uint32_t)set_count;
        alloc_info.pSetLayouts = layouts.data();
        alloc_info.descriptorPool = pool;

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        v.check_result(v.vkAllocateDescriptorSets(handles.device, &alloc_info, sets));
    }

    static void initialize_image(vulkan_context_t* context, VkCommandBuffer command_buffer,
                                 VkImage image) {
        ZoneScoped;

        VkImageMemoryBarrier image_barrier{};
        create_image_barrier(image_barrier, image, 0, image_access_flags, VK_IMAGE_LAYOUT_UNDEFINED,
                             image_compute_layout);

        context->vtable.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                             compute_stage, 0, 0, nullptr, 0, nullptr, 1,
                                             &image_barrier);
    }

    static void create_network_staging_buffer(vulkan_context_t* context, const network* nn,
                                              vulkan_buffer_t* buffer,
                                              std::vector<VkBufferImageCopy>& regions) {
        ZoneScoped;

        size_t total_size = 0;
        const auto& layers = nn->get_layers();
        for (const auto& layer : layers) {
            total_size += sizeof(number_t) * layer.size * (layer.previous_size + 1);
        }

        create_vulkan_buffer(context, total_size, buffer);

        const auto& v = context->vtable;
        const auto& handles = context->handles;

        number_t* mapped = nullptr;
        v.check_result(vmaMapMemory(handles.allocator, buffer->allocation, (void**)&mapped));

        size_t current_offset = 0;
        for (size_t i = 0; i < layers.size(); i++) {
            const auto& layer = layers[i];

            VkBufferImageCopy region{};
            region.bufferOffset = (VkDeviceSize)current_offset * sizeof(number_t);
            region.imageExtent.width = (uint32_t)(layer.previous_size + 1);
            region.imageExtent.height = (uint32_t)layer.size;
            region.imageExtent.depth = 1;
            region.imageOffset.z = (int32_t)i;
            region.imageSubresource.aspectMask = image_aspect_flags;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            regions.push_back(region);

            for (uint64_t c = 0; c < layer.size; c++) {
                mapped[current_offset] = layer.biases[c];
                copy(&layer.weights[c * layer.previous_size], &mapped[current_offset + 1],
                     layer.previous_size * sizeof(number_t));

                current_offset += layer.previous_size + 1;
            }
        }

        vmaUnmapMemory(handles.allocator, buffer->allocation);
    }

    static void fill_network_info(vulkan_layer_t* layers, const network* nn) { ZoneScoped; }

    void vulkan_evaluator::add_network_reference(const network* nn) {
        ZoneScoped;

        if (!m_network_data.contains(nn)) {
            auto& data = m_network_data[nn];
            data.references = 0;

            const auto& layers = nn->get_layers();
            size_t buffer_size = layers.size() * sizeof(vulkan_layer_t);

            VkExtent3D image_size{};
            image_size.depth = (uint32_t)layers.size();

            for (const auto& layer : layers) {
                image_size.width = std::max(image_size.width, (uint32_t)(layer.previous_size + 1));
                image_size.height = std::max(image_size.height, (uint32_t)layer.size);
            }

            create_vulkan_buffer(m_context.get(), buffer_size, &data.info_buffer);
            create_vulkan_image(m_context.get(), VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
                                image_size, &data.data_image);

            alloc_descriptor_sets(m_context.get(), m_objects.network_layout,
                                  m_objects.descriptor_pool, 1, &data.descriptor_set);

            const auto& v = m_context->vtable;
            const auto& handles = m_context->handles;

            vulkan_layer_t* layer_info = nullptr;
            v.check_result(vmaMapMemory(handles.allocator, data.info_buffer.allocation, (void**)&layer_info));

            for (size_t i = 0; i < layers.size(); i++) {
                const auto& layer = layers[i];
                auto& info = layer_info[i];

                info.size = (uint32_t)layer.size;
                info.previous_size = (uint32_t)layer.previous_size;
                info.activation_function = (uint32_t)layer.function;
            }

            vmaUnmapMemory(handles.allocator, data.info_buffer.allocation);

            VkDescriptorBufferInfo buffer_info{};
            buffer_info.range = (VkDeviceSize)buffer_size;
            buffer_info.offset = 0;
            buffer_info.buffer = data.info_buffer.buffer;

            VkDescriptorImageInfo image_info{};
            image_info.imageLayout = image_compute_layout;
            image_info.imageView = data.data_image.view;

            std::vector<VkWriteDescriptorSet> writes;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = data.descriptor_set;
            write.dstBinding = 0;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &buffer_info;
            writes.push_back(write);

            std::memset(&write, 0, sizeof(VkWriteDescriptorSet));
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = data.descriptor_set;
            write.dstBinding = 1;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.descriptorCount = 1;
            write.pImageInfo = &image_info;
            writes.push_back(write);

            v.vkUpdateDescriptorSets(handles.device, (uint32_t)writes.size(), writes.data(), 0,
                                     nullptr);

            VkImageMemoryBarrier src_barrier, dst_barrier;
            create_image_barrier(src_barrier, data.data_image.image, 0, transfer_dst_access,
                                 VK_IMAGE_LAYOUT_UNDEFINED, transfer_dst_layout);
            create_image_barrier(dst_barrier, data.data_image.image, transfer_dst_access,
                                 image_access_flags, transfer_dst_layout, image_compute_layout);

            vulkan_buffer_t staging_buffer;
            std::vector<VkBufferImageCopy> regions;
            create_network_staging_buffer(m_context.get(), nn, &staging_buffer, regions);

            VkCommandBuffer command_buffer =
                alloc_open_command_buffer(m_context.get(), m_objects.command_pool);

            v.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   transfer_stage, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

            v.vkCmdCopyBufferToImage(command_buffer, staging_buffer.buffer, data.data_image.image,
                                     transfer_dst_layout, (uint32_t)regions.size(), regions.data());

            v.vkCmdPipelineBarrier(command_buffer, transfer_stage, compute_stage, 0, 0, nullptr, 0,
                                   nullptr, 1, &dst_barrier);

            end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue, command_buffer,
                                          true, VK_NULL_HANDLE);

            v.vkFreeCommandBuffers(handles.device, m_objects.command_pool, 1, &command_buffer);
            destroy_vulkan_buffer(m_context.get(), &staging_buffer);
        }

        m_network_data[nn].references++;
    }

    void vulkan_evaluator::remove_network_reference(const network* nn) {
        ZoneScoped;

        auto& data = m_network_data[nn];
        if (--data.references == 0 && !is_training()) {
            const auto& v = m_context->vtable;
            const auto& handles = m_context->handles;

            v.vkFreeDescriptorSets(handles.device, m_objects.descriptor_pool, 1,
                                   &data.descriptor_set);

            destroy_vulkan_buffer(m_context.get(), &data.info_buffer);
            destroy_vulkan_image(m_context.get(), &data.data_image);

            m_network_data.erase(nn);
        }
    }

    void vulkan_evaluator::remove_pass_reference(uint64_t pass) {
        ZoneScoped;

        auto& data = m_passes[pass];
        if (--data.references == 0) {
            remove_network_reference(data.nn);

            const auto& v = m_context->vtable;
            const auto& handles = m_context->handles;

            v.check_result(v.vkFreeDescriptorSets(handles.device, m_objects.descriptor_pool, 1,
                                                  &data.descriptor_set));

            destroy_vulkan_image(m_context.get(), &data.activations);
            destroy_vulkan_image(m_context.get(), &data.z);
            destroy_vulkan_image(m_context.get(), &data.deltas);

            m_passes.erase(pass);
        }
    }

    uint64_t vulkan_evaluator::new_pass(const network* network,
                                        const std::vector<number_t>& inputs) {
        ZoneScoped;
        add_network_reference(network);
        const auto& layers = network->get_layers();

        uint64_t input_neurons = layers[0].previous_size;
        size_t input_count = inputs.size();

        uint64_t id = m_current_pass_id++;
        auto& pass = m_passes[id];

        pass.references = 1;
        pass.nn = network;
        pass.pass_id = id;
        pass.run_count = (input_count - (input_count % input_neurons)) / input_neurons;

        uint64_t max_neurons = 0;
        uint64_t max_neuron_size = 0;

        for (const auto& layer : layers) {
            max_neurons = std::max(max_neurons, layer.size);
            max_neuron_size = std::max(max_neuron_size, layer.previous_size + 1);
        }

        VkExtent3D activations_size{};
        activations_size.width = (uint32_t)std::max(max_neurons, layers[0].previous_size);
        activations_size.height = (uint32_t)layers.size() + 2;
        activations_size.depth = (uint32_t)pass.run_count;

        create_vulkan_image(m_context.get(), VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
                            activations_size, &pass.activations);

        VkExtent3D z_size{};
        z_size.width = (uint32_t)max_neurons;
        z_size.height = (uint32_t)layers.size();
        z_size.depth = (uint32_t)pass.run_count;

        create_vulkan_image(m_context.get(), VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, z_size,
                            &pass.z);

        const auto& network_data = m_network_data[network];
        auto delta_image_size = network_data.data_image.size;

        delta_image_size.depth *= (uint32_t)pass.run_count;
        create_vulkan_image(m_context.get(), VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D,
                            delta_image_size, &pass.deltas);

        alloc_descriptor_sets(m_context.get(), m_objects.evaluation_layout,
                              m_objects.descriptor_pool, 1, &pass.descriptor_set);

        std::vector<vulkan_image_t*> descriptor_images = { &pass.activations, &pass.z,
                                                           &pass.deltas };

        std::vector<VkDescriptorImageInfo> image_info(descriptor_images.size());
        std::vector<VkWriteDescriptorSet> writes(descriptor_images.size());

        vulkan_buffer_t staging_buffer;
        create_vulkan_buffer(m_context.get(), inputs.size() * sizeof(number_t), &staging_buffer);

        const auto& v = m_context->vtable;
        const auto& handles = m_context->handles;

        void* mapped = nullptr;
        vmaMapMemory(handles.allocator, staging_buffer.allocation, &mapped);
        copy(inputs.data(), mapped, staging_buffer.size);
        vmaUnmapMemory(handles.allocator, staging_buffer.allocation);

        VkCommandBuffer command_buffer =
            alloc_open_command_buffer(m_context.get(), m_objects.command_pool);

        {
            TracyVkZoneTransient(handles.profiler_context, vk_zone, command_buffer,
                                 "Set up pass data", m_profiling_enabled);

            VkImageMemoryBarrier src_barrier, dst_barrier;
            create_image_barrier(src_barrier, pass.activations.image, 0, transfer_dst_access,
                                 VK_IMAGE_LAYOUT_UNDEFINED, transfer_dst_layout);

            create_image_barrier(dst_barrier, pass.activations.image, transfer_dst_access,
                                 image_access_flags, transfer_dst_layout, image_compute_layout);

            v.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   transfer_stage, 0, 0, nullptr, 0, nullptr, 1, &src_barrier);

            std::vector<VkBufferImageCopy> regions;
            for (size_t i = 0; i < pass.run_count; i++) {
                auto& region = regions.emplace_back();
                std::memset(&region, 0, sizeof(VkBufferImageCopy));

                region.imageExtent.width = (uint32_t)input_neurons;
                region.imageExtent.height = 1;
                region.imageExtent.depth = 1;
                region.imageOffset.z = (int32_t)i;
                region.bufferOffset = (VkDeviceSize)(i * input_neurons * sizeof(number_t));
                region.imageSubresource.aspectMask = image_aspect_flags;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageSubresource.mipLevel = 0;
            }

            v.vkCmdCopyBufferToImage(command_buffer, staging_buffer.buffer, pass.activations.image,
                                     transfer_dst_layout, (uint32_t)regions.size(), regions.data());

            v.vkCmdPipelineBarrier(command_buffer, transfer_stage, compute_stage, 0, 0, nullptr, 0,
                                   nullptr, 1, &dst_barrier);

            for (size_t i = 0; i < descriptor_images.size(); i++) {
                auto image = descriptor_images[i];
                if (image != &pass.activations) {
                    initialize_image(m_context.get(), command_buffer, image->image);
                }

                auto& info = image_info[i];
                info.sampler = VK_NULL_HANDLE;
                info.imageLayout = image_compute_layout;
                info.imageView = image->view;

                auto& write = writes[i];
                std::memset(&write, 0, sizeof(VkWriteDescriptorSet));

                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write.dstSet = pass.descriptor_set;
                write.dstBinding = (uint32_t)i;
                write.dstArrayElement = 0;
                write.pImageInfo = &info;
            }
        }

        end_and_submit_command_buffer(m_context.get(), m_objects.compute_queue, command_buffer,
                                      true, VK_NULL_HANDLE);

        v.vkFreeCommandBuffers(handles.device, m_objects.command_pool, 1, &command_buffer);
        v.vkUpdateDescriptorSets(handles.device, (uint32_t)writes.size(), writes.data(), 0,
                                 nullptr);

        destroy_vulkan_buffer(m_context.get(), &staging_buffer);
        return id;
    }
} // namespace neuralnet::evaluators