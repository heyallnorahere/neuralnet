#include "nnpch.h"
#include "neuralnet/network.h"

namespace neuralnet {
    number_t& network::get_bias_address(layer_t& layer, uint64_t current) {
        return layer.biases[current];
    }

    number_t network::get_bias(const layer_t& layer, uint64_t current) {
        return layer.biases[current];
    }

    number_t& network::get_weight_address(layer_t& layer, uint64_t current, uint64_t previous) {
        // see neuralnet_layer_t::weights in network.h
        uint64_t index = current * layer.previous_size + previous;
        return layer.weights[index];
    }

    number_t network::get_weight(const layer_t& layer, uint64_t current, uint64_t previous) {
        uint64_t index = current * layer.previous_size + previous;
        return layer.weights[index];
    }

    void network::copy_layer(const layer_t& layer, layer_t& result) {
        result.size = layer.size;
        result.previous_size = layer.previous_size;
        result.function = layer.function;

        size_t biases_size = result.size * sizeof(number_t);
        size_t weights_size = biases_size * result.previous_size;

        result.biases = (number_t*)alloc(biases_size);
        result.weights = (number_t*)alloc(weights_size);

        copy(layer.biases, result.biases, biases_size);
        copy(layer.weights, result.weights, weights_size);
    }

    network::network(const std::vector<layer_t>& layers,
                     const std::vector<activation_function_t>& activations) {
        m_activation_functions = activations;

        for (size_t i = 0; i < layers.size(); i++) {
            const layer_t& src_layer = layers[i];
            if (src_layer.function >= m_activation_functions.size()) {
                throw std::runtime_error("invalid function!");
            }

            if (i > 0) {
                const layer_t& previous_layer = layers[i - 1];
                if (src_layer.previous_size != previous_layer.size) {
                    throw std::runtime_error("layer size mismatch!");
                }
            }

            layer_t& dst_layer = m_layers.emplace_back();
            copy_layer(src_layer, dst_layer);
        }
    }

    network::~network() {
        for (layer_t& layer : m_layers) {
            freemem(layer.biases);
            freemem(layer.weights);
        }
    }
} // namespace neuralnet