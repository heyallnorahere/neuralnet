#include <memory>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

#include <neuralnet.h>
#include <zlib.h>
using number_t = neuralnet::number_t;

static neuralnet::network* create_network(const std::vector<uint64_t>& layer_sizes,
                                          const neuralnet::activation_function_t& function) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<number_t> dist(-1, 1);

    std::vector<neuralnet::layer_t> layers;
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        auto& layer = layers.emplace_back();
        layer.size = layer_sizes[i + 1];
        layer.previous_size = layer_sizes[i];
        layer.function = 0;

        size_t biases_size = layer.size * sizeof(number_t);
        size_t weights_size = biases_size * layer.previous_size;

        layer.biases = (number_t*)neuralnet::alloc(biases_size);
        layer.weights = (number_t*)neuralnet::alloc(weights_size);

        for (size_t c = 0; c < layer.size; c++) {
            neuralnet::network::get_bias_address(layer, c) = dist(rng);

            for (size_t p = 0; p < layer.previous_size; p++) {
                neuralnet::network::get_weight_address(layer, c, p) = dist(rng);
            }
        }
    }

    auto network = new neuralnet::network(layers, { function });
    for (const auto& layer : layers) {
        neuralnet::freemem(layer.weights);
        neuralnet::freemem(layer.biases);
    }

    return network;
}

static number_t sigmoid(number_t x) { return 1 / (1 + exp(-x)); }
static number_t sigmoid_prime(number_t x) {
    number_t sig = sigmoid(x);
    return sig * (1 - sig);
}

static number_t string_to_number(const std::string& string) {
    switch (sizeof(number_t)) {
    case sizeof(float):
        return (number_t)std::stof(string);
    case sizeof(double):
        return (number_t)std::stod(string);
    case sizeof(long double):
        return (number_t)std::stold(string);
    }

    return 0;
}

static const std::vector<uint64_t> s_layer_sizes = { 2, 2, 2, 2 };
int main(int argc, const char** argv) {
    neuralnet::activation_function_t function;
    function.get = sigmoid;
    function.get_derivative = sigmoid_prime;

    auto evaluator = std::unique_ptr<neuralnet::evaluator>(neuralnet::create_cpu_evaluator());
    auto network = std::unique_ptr<neuralnet::network>(create_network(s_layer_sizes, function));

    int argument_count = argc - 1;
    std::vector<number_t> inputs(argument_count);
    for (int i = 0; i < argument_count; i++) {
        inputs[i] = string_to_number(argv[i + 1]);
    }

    auto key = evaluator->begin_eval(network.get(), inputs);
    if (!key) {
        std::cerr << "Failed to begin evaluation!" << std::endl;
        return 1;
    }

    uint64_t eval_key = key.value();
    while (!evaluator->is_result_ready(eval_key)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void* eval_output;
    if (!evaluator->get_eval_result(eval_key, &eval_output)) {
        std::cerr << "Failed to retrieve eval results pointer!" << std::endl;
        return 1;
    }

    std::vector<number_t> outputs;
    evaluator->retrieve_eval_values(network.get(), eval_output, outputs);

    for (size_t i = 0; i < outputs.size(); i++) {
        std::cout << "Output " << (i + 1) << ": " << outputs[i] << std::endl;
    }

    return 0;
}