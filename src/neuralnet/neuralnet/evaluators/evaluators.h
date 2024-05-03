#pragma once
#include "neuralnet/evaluator.h"

#ifdef NN_SUPPORT_vulkan
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
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
#define NN_DECLARE_VK_FUNCTION(name) PFN_##name name

    struct vulkan_vtable_t {
        vulkan_vtable_t() { std::memset(this, 0, sizeof(vulkan_vtable_t)); }

        NN_DECLARE_VK_FUNCTION(vkGetInstanceProcAddr);
        NN_DECLARE_VK_FUNCTION(vkGetDeviceProcAddr);

    };

    struct vulkan_handles_t {
        bool use_provided_handles;

        VkInstance instance;
        VkPhysicalDevice physical_device;
        VkDevice device;

        uint32_t compute_queue_index;
        std::unordered_set<uint32_t> shared_queue_indices;
    };

    struct vulkan_context_t {
        vulkan_vtable_t vtable;
        vulkan_handles_t handles;
    };

    class NN_API vulkan_evaluator : public evaluator {
    public:
        static void set_next_context(const vulkan_context_t& context);

        
    };
#endif

    NN_API std::unique_ptr<evaluator> choose_evaluator();
} // namespace neuralnet::evaluators
