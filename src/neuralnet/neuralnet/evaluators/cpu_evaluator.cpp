#include "nnpch.h"
#include "neuralnet/util.h"
#include "neuralnet/evaluators/evaluators.h"

namespace neuralnet::evaluators {
    struct cpu_backprop_data_t {
        const cpu_result_t* eval_result;
        const backprop_data_t* backprop_input;
    };

    static number_t sigmoid(number_t x) { return 1 / (1 + std::exp(-x)); }
    static number_t dsigmoid_dx(number_t x) {
        number_t sig = sigmoid(x);
        return sig * (1 - sig);
    }

    static number_t C(number_t x, number_t y) { return std::pow(x - y, 2); }
    static number_t dC_dx(number_t x, number_t y) { return 2 * (x - y); }

    static number_t A(activation_function func, number_t x) {
        switch (func) {
        case activation_function::sigmoid:
            return sigmoid(x);
        default:
            throw std::runtime_error("invalid activation function!");
            return 0;
        }
    }

    static number_t dA_dz(activation_function func, number_t x) {
        switch (func) {
        case activation_function::sigmoid:
            return dsigmoid_dx(x);
        default:
            throw std::runtime_error("invalid activation function!");
            return 0;
        }
    }

    bool cpu_evaluator::is_result_ready(uint64_t result) const {
        ZoneScoped;
        return m_results.find(result) != m_results.end();
    }

    bool cpu_evaluator::free_result(uint64_t result) {
        ZoneScoped;

        if (!is_result_ready(result)) {
            return false;
        }

        auto& data = m_results[result];
        for (void* ptr : data.results) {
            switch (data.type) {
            case cpu_result_type::backprop:
                delete (layer_t*)ptr;
                break;
            default:
                freemem(ptr);
                break;
            }
        }

        m_results.erase(result);
        return true;
    }

    struct cpu_inputs_t {
        const number_t* data;
        size_t count;
    };

    std::optional<uint64_t> cpu_evaluator::begin_eval(const network* nn,
                                                      const std::vector<number_t>& inputs) {
        ZoneScoped;

        const auto& layers = nn->get_layers();
        if (layers.empty()) {
            return {};
        }

        cpu_inputs_t input_data;
        input_data.data = inputs.data();
        input_data.count = inputs.size();

        return begin_eval(nn, &input_data);
    }

    std::optional<uint64_t> cpu_evaluator::begin_eval(const network* nn, void* native_inputs) {
        ZoneScoped;

        const auto& layers = nn->get_layers();
        if (layers.empty()) {
            return {};
        }

        uint64_t key = m_key++;
        auto& result = m_results[key];

        auto inputs = (cpu_inputs_t*)native_inputs;
        uint64_t input_count = layers[0].size;
        size_t pass_count = (inputs->count - (inputs->count % input_count)) / input_count;

        result.type = cpu_result_type::eval;
        result.nn = nn;
        result.passes = pass_count;

        for (size_t i = 0; i < pass_count; i++) {
            eval(&inputs->data[i * layers.size()], result);
        }

        return key;
    }

    bool cpu_evaluator::get_eval_result(uint64_t result, void** outputs) {
        ZoneScoped;

        if (!is_result_ready(result)) {
            return false;
        }

        auto& cpu_result = m_results[result];
        if (cpu_result.type != cpu_result_type::eval) {
            return false;
        }

        *outputs = &cpu_result; // not much more specific you can be in this case
        return true;
    }

    void cpu_evaluator::retrieve_eval_values(const network* nn, void* native_outputs,
                                             std::vector<number_t>& outputs) {
        ZoneScoped;

        const auto& layers = nn->get_layers();
        const auto& output_layer = layers[layers.size() - 1];

        auto result = (cpu_result_t*)native_outputs;
        void* layer_data = result->results[layers.size()];

        outputs.resize(output_layer.size);
        copy(layer_data, outputs.data(), output_layer.size * sizeof(number_t));
    }

    std::optional<uint64_t> cpu_evaluator::begin_backprop(const network* nn,
                                                          const backprop_data_t& data) {
        ZoneScoped;

        const auto& layers = nn->get_layers();
        if (layers.empty() || data.eval_outputs == nullptr) {
            return {};
        }

        auto eval_result = (cpu_result_t*)data.eval_outputs;
        if (eval_result->type != cpu_result_type::eval || eval_result->nn != nn) {
            return {};
        }

        uint64_t key = m_key++;
        auto& result = m_results[key];

        result.type = cpu_result_type::backprop;
        result.nn = nn;
        result.passes = eval_result->passes;

        cpu_backprop_data_t backprop_data;
        backprop_data.backprop_input = &data;
        backprop_data.eval_result = eval_result;

        for (size_t i = 0; i < result.passes; i++) {
            backprop(backprop_data, result, i * layers.size());
        }

        return key;
    }

    bool cpu_evaluator::compose_deltas(const delta_composition_data_t& data) {
        ZoneScoped;
        for (uint64_t key : data.backprop_keys) {
            if (!m_results.contains(key)) {
                return false;
            }
        }

        auto& layers = data.nn->get_layers();
        for (uint64_t key : data.backprop_keys) {
            const auto& result = m_results.at(key);
            if (result.nn != data.nn) {
                throw std::runtime_error("network mismatch!");
            }

            for (size_t i = 0; i < layers.size(); i++) {
                auto& layer = layers[i];
                for (size_t j = 0; j < result.passes; j++) {
                    auto delta = (layer_t*)result.results[j * layers.size() + i];
                    if (delta->size != layer.size || delta->previous_size != layer.previous_size) {
                        throw std::runtime_error("delta/layer size mismatch!");
                    }

                    for (size_t c = 0; c < layer.size; c++) {
                        number_t& bias = network::get_bias_address(layer, c);
                        number_t bias_delta = network::get_bias(*delta, c);
                        bias -= bias_delta * data.delta_scalar;

                        for (size_t p = 0; p < layer.previous_size; p++) {
                            number_t& weight = network::get_weight_address(layer, c, p);
                            number_t weight_delta = network::get_weight(*delta, c, p);
                            weight -= weight_delta * data.delta_scalar;
                        }
                    }
                }
            }
        }

        return true;
    }

    number_t cpu_evaluator::cost_function(number_t actual, number_t expected) const {
        return C(actual, expected);
    }

    void cpu_evaluator::eval(const number_t* inputs, cpu_result_t& result) {
        ZoneScoped;
        const auto& layers = result.nn->get_layers();

        std::vector<std::vector<number_t>> activations, z;
        activations.resize(layers.size());
        z.resize(layers.size());

        for (size_t i = 0; i < layers.size(); i++) {
            const auto& layer = layers[i];

            auto& layer_activations = activations[i];
            auto& layer_z = z[i];

            layer_activations.resize(layer.size);
            layer_z.resize(layer.size);

            auto previous_activations = i > 0 ? activations[i - 1].data() : inputs;
            for (uint64_t c = 0; c < layer.size; c++) {
                number_t neuron_z = layer.biases[c];

                for (uint64_t p = 0; p < layer.previous_size; p++) {
                    number_t weight = network::get_weight(layer, c, p);
                    number_t previous_activation = previous_activations[p];

                    neuron_z = weight * previous_activation;
                }

                layer_z[c] = neuron_z;
                layer_activations[c] = A(layer.function, neuron_z);
            }
        }

        size_t result_count = layers.size() + 1;
        for (size_t i = 0; i < result_count; i++) {
            if (i == 0) {
                size_t first_layer_size = layers[0].previous_size * sizeof(number_t);
                void* first_layer = alloc(first_layer_size);

                copy(inputs, first_layer, first_layer_size);
                result.results.push_back(first_layer);

                continue;
            }

            size_t layer_index = i - 1;
            auto& layer = layers[layer_index];

            size_t layer_size = layer.size * sizeof(number_t);
            size_t data_size = layer_size * 2;
            void* layer_data = alloc(data_size);

            // see cpu_result_t::results
            copy(activations[layer_index].data(), layer_data, layer_size);
            copy(z[layer_index].data(), (void*)((size_t)layer_data + layer_size), layer_size);

            result.results.push_back(layer_data);
        }
    }

    void cpu_evaluator::backprop(const cpu_backprop_data_t& data, cpu_result_t& result,
                                 size_t offset) {
        ZoneScoped;
        size_t first_index = result.results.size();

        const auto& layers = result.nn->get_layers();
        for (int64_t i = layers.size() - 1; i >= 0; i--) {
            const auto& layer = layers[i];

            auto delta = new layer_t;
            delta->size = layer.size;
            delta->previous_size = layer.previous_size;
            delta->biases.resize(delta->size);
            delta->weights.resize(delta->size * delta->previous_size);

            size_t result_offset = offset + i;
            auto layer_data = (number_t*)data.eval_result->results[result_offset + 1];
            auto previous_layer_data = (number_t*)data.eval_result->results[result_offset];

            for (uint64_t c = 0; c < layer.size; c++) {
                number_t z = layer_data[layer.size + c];

                number_t dC_da;
                if (i == layers.size() - 1) {
                    dC_da = dC_dx(data.backprop_input->expected_outputs[c], layer_data[c]);
                } else {
                    dC_da = 0;

                    size_t next_layer_index = i + 1;
                    const auto& next_layer = layers[next_layer_index];
                    const auto& next_delta = *(const layer_t*)result.results[first_index];

                    for (size_t n = 0; n < next_layer.size; n++) {
                        number_t weight = network::get_weight(next_layer, n, c);
                        number_t dC_db_n = network::get_bias(next_delta, n);
                        number_t dC_dz_n = dC_db_n / 1.f; // dz/db

                        dC_da += weight * dC_dz_n;
                    }
                }

                number_t dC_dz = dC_da * dA_dz(layer.function, z);
                network::get_bias_address(*delta, c) = dC_dz * 1.f; // dz/db

                for (uint64_t p = 0; p < layer.previous_size; p++) {
                    number_t previous_activation = previous_layer_data[p];
                    network::get_weight_address(*delta, c, p) = dC_dz * previous_activation;
                }
            }

            auto it = result.results.begin();
            std::advance(it, first_index);
            result.results.insert(it, delta);
        }
    }
} // namespace neuralnet::evaluators