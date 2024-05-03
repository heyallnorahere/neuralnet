#include "nnpch.h"
#include "neuralnet/evaluators/evaluators.h"
#include "neuralnet/util.h"

namespace neuralnet::evaluators {
    std::unique_ptr<evaluator> choose_evaluator() {
        ZoneScoped;

#ifdef NN_SUPPORT_cpu
        return unique(new cpu_evaluator);
#else
        return std::unique_ptr<evaluator>(); // empty
#endif
    }
}