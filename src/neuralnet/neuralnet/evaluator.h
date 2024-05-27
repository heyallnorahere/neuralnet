#pragma once
#include "neuralnet/network.h"

namespace neuralnet {
    struct backprop_data_t {
        void* eval_outputs;
        std::vector<number_t> expected_outputs;
    };

    struct delta_composition_data_t {
        network* nn;
        std::vector<uint64_t> backprop_keys;

        // value to scale all delta weights & biases by
        // if this value is negative, network will regress
        float delta_scalar;

        // if this is false, do not copy to canonical layer data
        // note: in some implementations, this will do nothing
        bool copy;
    };

    enum class evaluator_type { cpu, vulkan, other };

    class NN_API evaluator {
    public:
        evaluator() { m_training = false; }
        virtual ~evaluator() = default;

        bool is_training() const { return m_training; }
        void set_training(bool training) { m_training = training; }

        virtual evaluator_type get_type() const { return evaluator_type::other; }

        // checks if the requested result has finished computing
        virtual bool is_result_ready(uint64_t result) const = 0;

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
        // if asynchronous, the implementation must NOT reference the output from get_eval_result
        // during evaluation
        virtual std::optional<uint64_t> begin_backprop(const network* nn,
                                                       const backprop_data_t& data) = 0;

        // composes deltas from the evaluator's memory into the canonical neural network layers
        virtual bool compose_deltas(const delta_composition_data_t& data) = 0;

        // cost function for training
        virtual number_t cost_function(number_t actual, number_t expected) const = 0;

    private:
        bool m_training;
    };
} // namespace neuralnet