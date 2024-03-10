#include "nnpch.h"
#include "neuralnet/network.h"

namespace neuralnet {
    number_t* network::get_bias(layer_t* layer, uint64_t current) { return &layer->biases[current]; }

    number_t* network::get_weight(layer_t* layer, uint64_t current, uint64_t previous) {
        uint64_t index = current * layer->previous_size + previous;
        return &layer->weights[index];
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
            dst_layer.size = src_layer.size;
            dst_layer.previous_size = src_layer.previous_size;
            dst_layer.function = src_layer.function;

            size_t biases_size = dst_layer.size * sizeof(number_t);
            size_t weights_size = biases_size * dst_layer.previous_size;

            dst_layer.biases = (number_t*)alloc(biases_size);
            dst_layer.weights = (number_t*)alloc(weights_size);

            copy(src_layer.biases, dst_layer.biases, biases_size);
            copy(src_layer.weights, dst_layer.weights, weights_size);
        }
    }

    network::~network() {
        for (layer_t& layer : m_layers) {
            freemem(layer.biases);
            freemem(layer.weights);
        }
    }
} // namespace neuralnet