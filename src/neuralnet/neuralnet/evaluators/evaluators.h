#pragma once
#include "neuralnet/evaluator.h"

#ifdef NN_SUPPORT_vulkan
#define VK_NO_PROTOTYPES
#define TRACY_VK_USE_SYMBOL_TABLE

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <tracy/TracyVulkan.hpp>

#define NN_DECLARE_VK_FUNCTION(name) PFN_##name name
#endif

namespace neuralnet::evaluators {
#ifdef NN_SUPPORT_cpu
    enum cpu_result_type { eval, backprop };

    struct cpu_result_t {
        cpu_result_type type;
        const network* nn;

        // for eval, this vector would contain the inputs for the first element
        // after the first element, each pointer contains activations, and then pre-activations
        // for backprop, this vector contains deltas to apply to the neural network, typed layer_t
        std::vector<void*> results;
    };

    struct cpu_backprop_data_t;
    class NN_API cpu_evaluator : public evaluator {
    public:
        cpu_evaluator() { m_key = 0; }
        virtual ~cpu_evaluator() override = default;

        virtual bool is_result_ready(uint64_t result) override;
        virtual bool free_result(uint64_t result) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   const std::vector<number_t>& inputs) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn, void* native_inputs) override;

        virtual bool get_eval_result(uint64_t result, void** outputs) override;
        virtual void retrieve_eval_values(const network* nn, void* native_outputs,
                                          std::vector<number_t>& outputs) override;

        virtual std::optional<uint64_t> begin_backprop(const network* nn,
                                                       const backprop_data_t& data) override;

        virtual bool get_backprop_result(uint64_t result, std::vector<layer_t>& deltas) override;

        virtual void flush() override;
        virtual number_t cost_function(number_t actual, number_t expected) override;

    private:
        void eval(number_t* inputs, cpu_result_t& result);
        void backprop(const cpu_backprop_data_t& data, cpu_result_t& result);

        uint64_t m_key;
        std::unordered_map<uint64_t, cpu_result_t> m_results;
    };
#endif

#ifdef NN_SUPPORT_vulkan
    struct vulkan_vtable_t {
        // no struct vtable, memset is fine
        vulkan_vtable_t() { std::memset(this, 0, sizeof(vulkan_vtable_t)); }

        void (*check_result)(VkResult result);
        PFN_vkDebugUtilsMessengerCallbackEXT debug_callback;

        // vkGetXXXXXProcAddr
        NN_DECLARE_VK_FUNCTION(vkGetInstanceProcAddr);
        NN_DECLARE_VK_FUNCTION(vkGetDeviceProcAddr);

        // global functions
        NN_DECLARE_VK_FUNCTION(vkEnumerateInstanceExtensionProperties);
        NN_DECLARE_VK_FUNCTION(vkEnumerateInstanceLayerProperties);
        NN_DECLARE_VK_FUNCTION(vkCreateInstance);

        // instance functions
        NN_DECLARE_VK_FUNCTION(vkDestroyInstance);
        NN_DECLARE_VK_FUNCTION(vkEnumeratePhysicalDevices);
        NN_DECLARE_VK_FUNCTION(vkEnumerateDeviceExtensionProperties);
        NN_DECLARE_VK_FUNCTION(vkCreateDevice);

        // debug messenger
        NN_DECLARE_VK_FUNCTION(vkCreateDebugUtilsMessengerEXT);
        NN_DECLARE_VK_FUNCTION(vkDestroyDebugUtilsMessengerEXT);

        // creating objects
        NN_DECLARE_VK_FUNCTION(vkCreateDescriptorSetLayout);
        NN_DECLARE_VK_FUNCTION(vkCreateDescriptorPool);
        NN_DECLARE_VK_FUNCTION(vkAllocateDescriptorSets);
        NN_DECLARE_VK_FUNCTION(vkCreatePipelineLayout);
        NN_DECLARE_VK_FUNCTION(vkCreateComputePipelines);
        NN_DECLARE_VK_FUNCTION(vkCreateCommandPool);
        NN_DECLARE_VK_FUNCTION(vkAllocateCommandBuffers);
        NN_DECLARE_VK_FUNCTION(vkCreateFence);

        // destroying objects
        NN_DECLARE_VK_FUNCTION(vkDestroyFence);
        NN_DECLARE_VK_FUNCTION(vkFreeCommandBuffers);
        NN_DECLARE_VK_FUNCTION(vkDestroyCommandPool);
        NN_DECLARE_VK_FUNCTION(vkDestroyPipeline);
        NN_DECLARE_VK_FUNCTION(vkDestroyPipelineLayout);
        NN_DECLARE_VK_FUNCTION(vkFreeDescriptorSets);
        NN_DECLARE_VK_FUNCTION(vkDestroyDescriptorPool);
        NN_DECLARE_VK_FUNCTION(vkDestroyDescriptorSetLayout);
        NN_DECLARE_VK_FUNCTION(vkDestroyDevice);

        // sync/queues
        NN_DECLARE_VK_FUNCTION(vkBeginCommandBuffer);
        NN_DECLARE_VK_FUNCTION(vkEndCommandBuffer);
        NN_DECLARE_VK_FUNCTION(vkQueueSubmit);
        NN_DECLARE_VK_FUNCTION(vkGetFenceStatus);
        NN_DECLARE_VK_FUNCTION(vkWaitForFences);
        NN_DECLARE_VK_FUNCTION(vkResetFences);

        // commands
        NN_DECLARE_VK_FUNCTION(vkCmdPipelineBarrier);
        NN_DECLARE_VK_FUNCTION(vkCmdBindPipeline);
        NN_DECLARE_VK_FUNCTION(vkCmdBindDescriptorSets);
        NN_DECLARE_VK_FUNCTION(vkCmdDispatch);
    };

    struct vulkan_handles_t {
        vulkan_handles_t() {
            vulkan_version = 0;
            context_provided = false;
            profiler_context = nullptr;
        }

        bool context_provided;
        uint32_t vulkan_version;

        VkInstance instance;
        VkDebugUtilsMessengerEXT debug_messenger;
        VkPhysicalDevice physical_device;
        VkDevice device;
        VmaAllocator allocator;

        TracyVkCtx profiler_context;

        uint32_t compute_queue_index;
        std::unordered_set<uint32_t> shared_queue_indices;
    };

    struct vulkan_context_t {
        vulkan_vtable_t vtable;
        vulkan_handles_t handles;
    };

    class NN_API vulkan_evaluator : public evaluator {
    public:
        static void set_next_context(std::unique_ptr<vulkan_context_t>&& context);
        static bool is_context_valid();

        vulkan_evaluator();
        virtual ~vulkan_evaluator() override;

        virtual bool is_result_ready(uint64_t result) override;
        virtual bool free_result(uint64_t result) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   const std::vector<number_t>& inputs) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn, void* native_inputs) override;

        virtual bool get_eval_result(uint64_t result, void** outputs) override;
        virtual void retrieve_eval_values(const network* nn, void* native_outputs,
                                          std::vector<number_t>& outputs) override;

        virtual std::optional<uint64_t> begin_backprop(const network* nn,
                                                       const backprop_data_t& data) override;

        virtual bool get_backprop_result(uint64_t result, std::vector<layer_t>& deltas) override;

        virtual void flush() override;
        virtual number_t cost_function(number_t actual, number_t expected) override;

    private:
        std::unique_ptr<vulkan_context_t> m_context;
    };
#endif

    NN_API std::unique_ptr<evaluator> choose_evaluator();
} // namespace neuralnet::evaluators
