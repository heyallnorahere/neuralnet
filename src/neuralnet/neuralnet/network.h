#pragma once

#if defined(NN_PCH_INCLUDED) && defined(__cplusplus)
using neuralnet_number_t = neuralnet::number_t;
#else
using neuralnet_number_t = double;
#endif

typedef struct {
    neuralnet_number_t(*get)(neuralnet_number_t x);
    neuralnet_number_t(*get_derivative)(neuralnet_number_t x);
} neuralnet_activation_function_t;

typedef struct {
    uint64_t size;
    uint64_t previous_size;
    size_t function;

    neuralnet_number_t* biases;
    neuralnet_number_t* weights; // laid out row to row; rows represents neurons on the current layer
} neuralnet_layer_t;

#ifdef __cplusplus

namespace neuralnet {
    using layer_t = neuralnet_layer_t;
    using activation_function_t = neuralnet_activation_function_t;

#ifndef NN_PCH_INCLUDED
    using number_t = neuralnet_number_t;
#endif

    class NN_API network {
    public:
        static number_t& get_bias_address(layer_t& layer, uint64_t current);
        static number_t get_bias(const layer_t& layer, uint64_t current);
        static number_t& get_weight_address(layer_t& layer, uint64_t current, uint64_t previous);
        static number_t get_weight(const layer_t& layer, uint64_t current, uint64_t previous);
        static void copy_layer(const layer_t& layer, layer_t& result);

        network(const std::vector<layer_t>& layers, const std::vector<activation_function_t>& activations);
        ~network();

        network(const network&) = delete;
        network& operator=(const network&) = delete;

        std::vector<layer_t>& get_layers() { return m_layers; }
        const std::vector<layer_t>& get_layers() const { return m_layers; }

        std::vector<activation_function_t>& get_activation_functions() { return m_activation_functions; }
        const std::vector<activation_function_t>& get_activation_functions() const { return m_activation_functions; }

    private:
        std::vector<layer_t> m_layers;
        std::vector<activation_function_t> m_activation_functions;
    };
} // namespace neuralnet

#endif