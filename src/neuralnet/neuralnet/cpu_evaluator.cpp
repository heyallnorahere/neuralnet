#include "nnpch.h"
#include "neuralnet/evaluator.h"

namespace neuralnet {
    enum cpu_result_type { eval, backprop };

    struct cpu_result_t {
        cpu_result_type type;
        const network* nn;

        // for eval, this vector would contain the inputs for the first element
        // after the first element, each pointer contains activations, and then pre-activations
        // for backprop, this vector contains deltas to apply to the neural network, typed layer_t
        std::vector<void*> results;
    };

    class cpu_evaluator : public evaluator {
    public:
        cpu_evaluator() { m_key = 0; }

        virtual ~cpu_evaluator() override = default;

        virtual bool is_result_ready(uint64_t result) override {
            return m_results.find(result) != m_results.end();
        }

        virtual bool free_result(uint64_t result) override {
            if (!is_result_ready(result)) {
                return false;
            }

            auto& data = m_results[result];
            for (void* ptr : data.results) {
                freemem(ptr);
            }

            m_results.erase(result);
            return true;
        }

        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   const std::vector<number_t>& inputs) override {
            const auto& layers = nn->get_layers();
            if (layers.empty() || inputs.size() != layers[0].previous_size) {
                return {};
            }

            return begin_eval(nn, (void*)inputs.data());
        }

        virtual std::optional<uint64_t> begin_eval(const network* nn,
                                                   void* native_inputs) override {
            const auto& layers = nn->get_layers();
            if (layers.empty()) {
                return {};
            }

            uint64_t key = m_key++;
            auto& result = m_results[key];

            result.type = cpu_result_type::eval;
            result.nn = nn;
            eval((number_t*)native_inputs, result);

            return key;
        }

        virtual bool get_eval_result(uint64_t result, void** outputs) override {
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

        virtual void retrieve_eval_values(const network* nn, void* native_outputs,
                                          std::vector<number_t>& outputs) override {
            const auto& layers = nn->get_layers();
            const auto& output_layer = layers[layers.size() - 1];

            auto result = (cpu_result_t*)native_outputs;
            void* layer_data = result->results[layers.size()];

            outputs.resize(output_layer.size);
            copy(layer_data, outputs.data(), output_layer.size * sizeof(number_t));
        }

        virtual std::optional<uint64_t> begin_backprop(const network* nn, void* outputs) override {
            return {};
        }

        virtual bool get_backprop_result(uint64_t result, std::vector<layer_t>& deltas) override {
            return false;
        }

    private:
        void eval(number_t* inputs, cpu_result_t& result) {
            const auto& layers = result.nn->get_layers();
            const auto& functions = result.nn->get_activation_functions();

            std::vector<std::vector<number_t>> activations, z;
            activations.resize(layers.size());
            z.resize(layers.size());

            for (size_t i = 0; i < layers.size(); i++) {
                const auto& layer = layers[i];
                const auto& function = functions[layer.function];

                auto& layer_activations = activations[i];
                auto& layer_z = z[i];

                number_t* previous_activations = i > 0 ? activations[i - 1].data() : inputs;
                for (size_t c = 0; c < layer.size; c++) {
                    number_t neuron_z = layer.biases[c];

                    for (size_t p = 0; p < layer.previous_size; p++) {
                        // see neuralnet_layer_t::weights in network.h
                        uint64_t weight_index = c * layer.previous_size + p;

                        number_t weight = layer.weights[weight_index];
                        number_t previous_activation = previous_activations[p];

                        neuron_z = weight * previous_activation;
                    }

                    layer_z[c] = neuron_z;
                    layer_activations[c] = function.get(neuron_z);
                }
            }

            size_t result_count = layers.size() + 1;
            result.results.resize(result_count);

            for (size_t i = 0; i < result_count; i++) {
                if (i == 0) {
                    size_t first_layer_size = layers[0].previous_size * sizeof(number_t);
                    void* first_layer = alloc(first_layer_size);

                    copy(inputs, first_layer, first_layer_size);
                    result.results[0] = first_layer;

                    continue;
                }

                size_t layer_index = i - 1;
                auto& layer = layers[layer_index];

                size_t layer_size = layer.size * sizeof(number_t);
                size_t data_size = layer_size * 2;
                void* layer_data = alloc(data_size);

                // see cpu_result_t::results
                copy(activations[layer_index].data(), layer_data, layer_size);
                copy(z[layer_index].data(), layer_data + layer_size, layer_size);

                result.results[i] = layer_data;
            }
        }

        uint64_t m_key;
        std::unordered_map<uint64_t, cpu_result_t> m_results;
    };

    evaluator* create_cpu_evaluator() { return new cpu_evaluator; }
} // namespace neuralnet