#include "nnpch.h"
#include "neuralnet/network.h"
#include "neuralnet/util.h"

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

    network* network::randomize(uint64_t input_size, const std::vector<layer_spec_t>& layers) {
        ZoneScoped;

        static constexpr number_t min = -1;
        static constexpr number_t max = 1;

        std::vector<layer_t> layer_data(layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            const auto& spec = layers[i];
            auto& layer = layer_data[i];

            layer.function = spec.function;
            layer.size = spec.size;
            layer.previous_size = i > 0 ? layers[i - 1].size : input_size;

            layer.biases.resize(layer.size);
            layer.weights.resize(layer.size * layer.previous_size);

            for (uint64_t c = 0; c < layer.size; c++) {
                layer.biases[c] = random::next(min, max);

                for (uint64_t p = 0; p < layer.previous_size; p++) {
                    layer.weights[c * layer.previous_size + p] = random::next(min, max);
                }
            }
        }

        return new network(layer_data);
    }

    network* network::randomize(const std::vector<uint64_t>& layer_sizes, activation_function function) {
        ZoneScoped;

        std::vector<layer_spec_t> layers;
        for (size_t i = 1; i < layer_sizes.size(); i++) {
            auto& spec = layers.emplace_back();
            spec.size = layer_sizes[i];
            spec.function = function;
        }

        return randomize(layer_sizes[0], layers);
    }

    void network::copy_layer(const layer_t& layer, layer_t& result) {
        ZoneScoped;

        result.size = layer.size;
        result.previous_size = layer.previous_size;
        result.function = layer.function;

        result.biases.resize(result.size);
        result.weights.resize(result.size * result.previous_size);

        copy(layer.biases.data(), result.biases.data(), result.biases.size() * sizeof(number_t));
        copy(layer.weights.data(), result.weights.data(), result.weights.size() * sizeof(number_t));
    }

    network::network(const std::vector<layer_t>& layers) {
        ZoneScoped;

        for (size_t i = 0; i < layers.size(); i++) {
            const layer_t& src_layer = layers[i];
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
        ZoneScoped;

        // nothing
    }
} // namespace neuralnet