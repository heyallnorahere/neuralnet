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
    enum class cpu_result_type { eval, backprop };

    struct cpu_result_t {
        cpu_result_type type;
        const network* nn;

        // for eval, this vector would contain the inputs for the first element
        // after the first element, each pointer contains activations, and then pre-activations
        // for backprop, this vector contains deltas to apply to the neural network, typed layer_t
        std::vector<void*> results;
        size_t passes;
    };

    struct cpu_backprop_data_t;
    class NN_API cpu_evaluator : public evaluator {
    public:
        cpu_evaluator() { m_key = 0; }
        virtual ~cpu_evaluator() override = default;

        virtual evaluator_type get_type() const override { return evaluator_type::cpu; }

        virtual bool is_result_ready(uint64_t result) const override;
        virtual bool free_result(uint64_t result) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   const std::vector<number_t>& inputs) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn, void* native_inputs) override;

        virtual bool get_eval_result(uint64_t result, void** outputs) override;
        virtual void retrieve_eval_values(const network* nn, void* native_outputs,
                                          std::vector<number_t>& outputs) override;

        virtual std::optional<uint64_t> begin_backprop(const network* nn,
                                                       const backprop_data_t& data) override;

        virtual bool compose_deltas(const delta_composition_data_t& data) override;

        virtual number_t cost_function(number_t actual, number_t expected) const override;

    private:
        void eval(const number_t* inputs, cpu_result_t& result);
        void backprop(const cpu_backprop_data_t& data, cpu_result_t& result, size_t offset);

        uint64_t m_key;
        std::unordered_map<uint64_t, cpu_result_t> m_results;
    };
#endif

#ifdef NN_SUPPORT_vulkan
    struct vulkan_context_t;
    using user_callback_t = void(*)(vulkan_context_t* context);

    struct vulkan_user_callbacks_t {
        void* user_data;
        user_callback_t device_chosen;
        user_callback_t init_finished;
    };

    struct vulkan_vtable_t {
        // no struct vtable, memset is fine
        vulkan_vtable_t() { std::memset(this, 0, sizeof(vulkan_vtable_t)); }

        void (*check_result)(VkResult result);
        PFN_vkDebugUtilsMessengerCallbackEXT debug_callback;

        VkAllocationCallbacks alloc_callbacks;
        vulkan_user_callbacks_t user_callbacks;

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
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceFeatures);
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceFeatures2);
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceProperties);
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceQueueFamilyProperties);
        NN_DECLARE_VK_FUNCTION(vkGetPhysicalDeviceImageFormatProperties);
        NN_DECLARE_VK_FUNCTION(vkEnumerateDeviceExtensionProperties);
        NN_DECLARE_VK_FUNCTION(vkCreateDevice);

        // debug messenger
        NN_DECLARE_VK_FUNCTION(vkCreateDebugUtilsMessengerEXT);
        NN_DECLARE_VK_FUNCTION(vkDestroyDebugUtilsMessengerEXT);

        // creating objects
        NN_DECLARE_VK_FUNCTION(vkCreateDescriptorSetLayout);
        NN_DECLARE_VK_FUNCTION(vkCreateDescriptorPool);
        NN_DECLARE_VK_FUNCTION(vkAllocateDescriptorSets);
        NN_DECLARE_VK_FUNCTION(vkCreateShaderModule);
        NN_DECLARE_VK_FUNCTION(vkCreatePipelineLayout);
        NN_DECLARE_VK_FUNCTION(vkCreateComputePipelines);
        NN_DECLARE_VK_FUNCTION(vkCreateCommandPool);
        NN_DECLARE_VK_FUNCTION(vkAllocateCommandBuffers);
        NN_DECLARE_VK_FUNCTION(vkCreateFence);
        NN_DECLARE_VK_FUNCTION(vkCreateImageView);

        // destroying objects
        NN_DECLARE_VK_FUNCTION(vkDestroyImageView);
        NN_DECLARE_VK_FUNCTION(vkDestroyFence);
        NN_DECLARE_VK_FUNCTION(vkFreeCommandBuffers);
        NN_DECLARE_VK_FUNCTION(vkDestroyCommandPool);
        NN_DECLARE_VK_FUNCTION(vkDestroyPipeline);
        NN_DECLARE_VK_FUNCTION(vkDestroyPipelineLayout);
        NN_DECLARE_VK_FUNCTION(vkDestroyShaderModule);
        NN_DECLARE_VK_FUNCTION(vkFreeDescriptorSets);
        NN_DECLARE_VK_FUNCTION(vkDestroyDescriptorPool);
        NN_DECLARE_VK_FUNCTION(vkDestroyDescriptorSetLayout);
        NN_DECLARE_VK_FUNCTION(vkDestroyDevice);

        // sync/queues
        NN_DECLARE_VK_FUNCTION(vkBeginCommandBuffer);
        NN_DECLARE_VK_FUNCTION(vkEndCommandBuffer);
        NN_DECLARE_VK_FUNCTION(vkGetDeviceQueue);
        NN_DECLARE_VK_FUNCTION(vkQueueSubmit);
        NN_DECLARE_VK_FUNCTION(vkQueueWaitIdle);
        NN_DECLARE_VK_FUNCTION(vkGetFenceStatus);
        NN_DECLARE_VK_FUNCTION(vkWaitForFences);
        NN_DECLARE_VK_FUNCTION(vkResetFences);

        // commands
        NN_DECLARE_VK_FUNCTION(vkCmdPipelineBarrier);
        NN_DECLARE_VK_FUNCTION(vkCmdBindPipeline);
        NN_DECLARE_VK_FUNCTION(vkCmdBindDescriptorSets);
        NN_DECLARE_VK_FUNCTION(vkCmdPushConstants);
        NN_DECLARE_VK_FUNCTION(vkCmdDispatch);
        NN_DECLARE_VK_FUNCTION(vkCmdCopyBufferToImage);
        NN_DECLARE_VK_FUNCTION(vkCmdCopyImageToBuffer);

        // idk man
        NN_DECLARE_VK_FUNCTION(vkUpdateDescriptorSets);
    };

    struct vulkan_handles_t {
        vulkan_handles_t() {
            vulkan_version = 0;
            additional_image_usage = 0;

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
        VkImageUsageFlags additional_image_usage;
        std::unordered_set<uint32_t> shared_queue_indices;
        std::unordered_set<std::string> instance_extensions, device_extensions;
    };

    struct vulkan_context_t {
        std::string name;
        vulkan_vtable_t vtable;
        vulkan_handles_t handles;
    };

    struct vulkan_evaluator_objects_t {
        VkQueue compute_queue;
        VkDescriptorPool descriptor_pool;
        VkCommandPool command_pool;

        VkDescriptorSetLayout evaluation_layout, network_layout;
        VkPipelineLayout pipeline_layout;
        std::unordered_map<std::string, VkPipeline> pipelines;
    };

    struct vulkan_buffer_t {
        VkBuffer buffer;
        VmaAllocation allocation;
        size_t size;
    };

    struct vulkan_image_t {
        VkImage image;
        VkImageView view;
        VmaAllocation allocation;

        VkExtent3D size;
        VkImageType type;
        VkImageViewType view_type;
    };

    struct vulkan_network_data_t {
        vulkan_buffer_t info_buffer;
        vulkan_image_t data_image;
        VkDescriptorSet descriptor_set;

        uint64_t references;
    };

    struct vulkan_pass_data_t {
        vulkan_image_t activations, z, deltas;
        VkDescriptorSet descriptor_set;

        uint64_t references, pass_id;
        size_t run_count;

        const network* nn;
    };

    enum class vulkan_result_type { eval, backprop };
    struct vulkan_result_t {
        vulkan_result_type type;
        uint64_t pass;

        VkCommandBuffer command_buffer;
        VkFence fence;

        std::vector<vulkan_buffer_t> staging_buffers;
    };

    class NN_API vulkan_evaluator : public evaluator {
    public:
        static void set_next_context(std::unique_ptr<vulkan_context_t>&& context);
        static bool is_context_valid();

        vulkan_evaluator();
        virtual ~vulkan_evaluator() override;

        virtual evaluator_type get_type() const override { return evaluator_type::vulkan; }

        virtual bool is_result_ready(uint64_t result) const override;
        virtual bool free_result(uint64_t result) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   const std::vector<number_t>& inputs) override;

        virtual std::optional<uint64_t> begin_eval(const network* nn, void* native_inputs) override;

        virtual bool get_eval_result(uint64_t result, void** outputs) override;
        virtual void retrieve_eval_values(const network* nn, void* native_outputs,
                                          std::vector<number_t>& outputs) override;

        virtual std::optional<uint64_t> begin_backprop(const network* nn,
                                                       const backprop_data_t& data) override;

        virtual bool compose_deltas(const delta_composition_data_t& data) override;

        virtual number_t cost_function(number_t actual, number_t expected) const override;

        vulkan_context_t* get_context();
        vulkan_network_data_t* get_network_data(const network* network);
        vulkan_pass_data_t* get_pass_data(uint64_t result);

    private:
        void init_vulkan();
        void shutdown_vulkan();

        void add_network_reference(const network* network);
        void remove_network_reference(const network* network);

        void remove_pass_reference(uint64_t pass);
        uint64_t new_pass(const network* network, const std::vector<number_t>& inputs);


        std::unique_ptr<vulkan_context_t> m_context;
        vulkan_evaluator_objects_t m_objects;
        bool m_profiling_enabled;

        uint64_t m_current_pass_id, m_current_result_id;
        std::unordered_map<const network*, vulkan_network_data_t> m_network_data;
        std::unordered_map<uint64_t, vulkan_result_t> m_results;
        std::unordered_map<uint64_t, vulkan_pass_data_t> m_passes;
    };
#endif

    NN_API bool is_evaluator_supported(evaluator_type type);
    NN_API evaluator_type get_preferred_evaluator();
    NN_API evaluator* choose_evaluator(evaluator_type preferred = evaluator_type::other);
} // namespace neuralnet::evaluators
