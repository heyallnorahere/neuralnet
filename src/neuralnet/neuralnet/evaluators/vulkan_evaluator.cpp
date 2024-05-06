#include "nnpch.h"
#include "neuralnet/evaluators/evaluators.h"
#include "neuralnet/util.h"

namespace neuralnet::evaluators {
    static std::unique_ptr<vulkan_context_t> s_next_context;

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

        NN_LOAD_VK_GLOBAL(vtable, vkEnumerateInstanceExtensionProperties);
        NN_LOAD_VK_GLOBAL(vtable, vkEnumerateInstanceLayerProperties);
        NN_LOAD_VK_GLOBAL(vtable, vkCreateInstance);
    }

    static void vtable_load_instance(vulkan_vtable_t& vtable, VkInstance instance) {
        ZoneScoped;
        NN_LOAD_VK_INSTANCE(vtable, instance, vkGetDeviceProcAddr);

        NN_LOAD_VK_INSTANCE(vtable, instance, vkDestroyInstance);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkEnumeratePhysicalDevices);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkEnumerateDeviceExtensionProperties);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkCreateDevice);

        NN_LOAD_VK_INSTANCE(vtable, instance, vkCreateDebugUtilsMessengerEXT);
        NN_LOAD_VK_INSTANCE(vtable, instance, vkDestroyDebugUtilsMessengerEXT);
    }

    static void vtable_load_device(vulkan_vtable_t& vtable, VkDevice device) {
        ZoneScoped;

        // todo: load device functions (a lot)
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

        

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.apiVersion = context->handles.vulkan_version;
        app_info.engineVersion = app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        app_info.pEngineName = app_info.pApplicationName = "Application using neuralnet";

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        // todo: is this necessary?
        create_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    static void create_debug_messenger(vulkan_context_t* context) {
        ZoneScoped;

        // todo: create debug messenger, if function is present
    }

    static void select_physical_device(vulkan_context_t* context) {
        ZoneScoped;

        // todo: select physical device
    }

    static void create_device(vulkan_context_t* context) {
        ZoneScoped;

        // todo: create device
    }

    static void init_vulkan(vulkan_context_t* context) {
        ZoneScoped;

        vtable_load_globals(context->vtable);
        if (!context->handles.context_provided) {
            if (context->handles.vulkan_version == 0) {
                // don't need any higher version than 1.0
                context->handles.vulkan_version = VK_VERSION_1_0;
            }

            create_instance(context);
        }

        vtable_load_instance(context->vtable, context->handles.instance);
        if (!context->handles.context_provided) {
            create_debug_messenger(context);

            select_physical_device(context);
            create_device(context);
        }

        vtable_load_device(context->vtable, context->handles.device);
        // todo: further init
    }

    vulkan_evaluator::vulkan_evaluator() {
        ZoneScoped;
        if (!is_context_valid()) {
            throw std::runtime_error("No valid context!");
        }

        m_context = std::move(s_next_context);
        init_vulkan(m_context.get());
    }

    vulkan_evaluator::~vulkan_evaluator() {
        ZoneScoped;

        // todo: destroy
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