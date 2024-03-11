#pragma once
#include "neuralnet/network.h"

namespace neuralnet {
    struct backprop_data_t {
        void* eval_outputs;
        std::vector<number_t> expected_outputs;
        number_t (*cost_derivative)(number_t x, number_t y);
    };

    class NN_API evaluator {
    public:
        virtual ~evaluator() = default;

        // checks if the requested result has finished computing
        virtual bool is_result_ready(uint64_t result) = 0;

        // frees comitted resources used by the requested result
        virtual bool free_result(uint64_t result) = 0;

        // begins evaluating the provided neural network with the provided inputs, in the form of a
        // number array
        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   const std::vector<number_t>& inputs) = 0;

        // begins evaluating the provided neural network with the provided inputs, in the form of a
        // native container
        virtual std::optional<uint64_t> begin_eval(const network* nn, void* native_inputs) = 0;

        // retrieves the native result of the evaluation requested
        virtual bool get_eval_result(uint64_t result, void** outputs) = 0;

        virtual void retrieve_eval_values(const network* nn, void* native_outputs,
                                          std::vector<number_t>& outputs) = 0;

        // begins performing backpropagation on the provided neural network, given previous
        // evaluation results
        virtual std::optional<uint64_t> begin_backprop(const network* nn,
                                                       const backprop_data_t& data) = 0;

        // retrieves deltas computed via gradient descent during backprop with the given key
        virtual bool get_backprop_result(uint64_t result, std::vector<layer_t>& deltas) = 0;
    };

    NN_API evaluator* create_cpu_evaluator();
} // namespace neuralnet