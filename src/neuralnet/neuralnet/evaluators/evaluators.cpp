#include "nnpch.h"
#include "neuralnet/evaluators/evaluators.h"
#include "neuralnet/util.h"

#ifdef NN_SUPPORT_vulkan
#include <volk.h>
#endif

namespace neuralnet::evaluators {
    std::unique_ptr<evaluator> choose_evaluator() {
        ZoneScoped;

#ifdef NN_SUPPORT_vulkan
        if (!vulkan_evaluator::is_context_valid()) {
            volkInitialize();

            auto context = std::make_unique<vulkan_context_t>();
            context->vtable.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
            context->vtable.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

            vulkan_evaluator::set_next_context(std::move(context));
        }

        return unique(new vulkan_evaluator);
#elif defined(NN_SUPPORT_cpu)
        return unique(new cpu_evaluator);
#else
        return std::unique_ptr<evaluator>(); // empty
#endif
    }
}