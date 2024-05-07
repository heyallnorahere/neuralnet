#include "nnpch.h"
#include "neuralnet/evaluators/evaluators.h"
#include "neuralnet/util.h"

#ifdef NN_SUPPORT_vulkan
#include <volk.h>
#endif

namespace neuralnet::evaluators {
    bool is_evaluator_supported(evaluator_type type) {
        ZoneScoped;
        switch (type) {
#ifdef NN_SUPPORT_vulkan
        case evaluator_type::vulkan:
            return true;
#endif
#ifdef NN_SUPPORT_cpu
        case evaluator_type::cpu:
            return true;
#endif
        default:
            return false;
        }
    }

    evaluator_type get_preferred_evaluator() {
        ZoneScoped;

#ifdef NN_SUPPORT_vulkan
        return evaluator_type::vulkan;
#elif defined(NN_SUPPORT_cpu)
        return evaluator_type::cpu;
#else
        return evaluator_type::other; // none available
#endif
    }

    evaluator* choose_evaluator(evaluator_type preferred) {
        ZoneScoped;

        if (preferred == evaluator_type::other) {
            preferred = get_preferred_evaluator();
        }

        switch (preferred) {
#ifdef NN_SUPPORT_vulkan
        case evaluator_type::vulkan:
            if (!vulkan_evaluator::is_context_valid()) {
                volkInitialize();

                auto context = std::make_unique<vulkan_context_t>();
                context->vtable.vkGetInstanceProcAddr = vkGetInstanceProcAddr;

                vulkan_evaluator::set_next_context(std::move(context));
            }

            return new vulkan_evaluator;
#endif
#ifdef NN_SUPPORT_cpu
        case evaluator_type::cpu:
            return new cpu_evaluator;
#endif
        default:
            return nullptr; // not supported/none available
        }
    }
} // namespace neuralnet::evaluators