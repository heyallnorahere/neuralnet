#pragma once

namespace neuralnet {
    enum class activation_function { sigmoid };

    struct layer_t {
        uint64_t size;
        uint64_t previous_size;
        activation_function function;

        std::vector<number_t> biases;
        std::vector<number_t> weights; // laid out row to row; rows represents neurons on the current layer
    };

    struct layer_spec_t {
        uint64_t size;
        activation_function function;
    };

    class NN_API network {
    public:
        static number_t& get_bias_address(layer_t& layer, uint64_t current);
        static number_t get_bias(const layer_t& layer, uint64_t current);
        static number_t& get_weight_address(layer_t& layer, uint64_t current, uint64_t previous);
        static number_t get_weight(const layer_t& layer, uint64_t current, uint64_t previous);

        static network* randomize(uint64_t input_size, const std::vector<layer_spec_t>& layers);
        static network* randomize(const std::vector<uint64_t>& layer_sizes, activation_function function);
        static void copy_layer(const layer_t& layer, layer_t& result);

        network(const std::vector<layer_t>& layers);
        ~network();

        network(const network&) = delete;
        network& operator=(const network&) = delete;

        std::vector<layer_t>& get_layers() { return m_layers; }
        const std::vector<layer_t>& get_layers() const { return m_layers; }

    private:
        std::vector<layer_t> m_layers;
    };
} // namespace neuralnet