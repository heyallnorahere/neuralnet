#pragma once

typedef double(*activation_function_t)(double x);
typedef struct {
    uint64_t size;
    uint64_t previous_size;
    uint32_t function;

    double* biases;
    double* weights; // laid out row to row; rows represents neurons on the current layer
} layer_t;

#ifdef __cplusplus

namespace neuralnet {
    class network {
    public:
        static double* get_bias(layer_t* layer, uint64_t current);
        static double* get_weight(layer_t* layer, uint64_t current, uint64_t previous);

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