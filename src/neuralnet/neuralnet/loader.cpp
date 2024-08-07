#include "nnpch.h"
#include "neuralnet/loader.h"
#include "neuralnet/util.h"
#include "neuralnet/compression.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace neuralnet {
    struct layer_desc_t {
        fs::path path;
        uint64_t size;
        activation_function function;
    };

    struct network_desc_t {
        uint64_t input_count;
        std::vector<layer_desc_t> layers;
    };

    void from_json(const json& src, layer_desc_t& dst) {
        ZoneScoped;

        src["path"].get_to(dst.path);
        src["size"].get_to(dst.size);

        static const std::unordered_map<std::string, activation_function> function_map = {
            { "sigmoid", activation_function::sigmoid }
        };

        auto function_name = src["function"].get<std::string>();
        dst.function = function_map.at(function_name);
    }

    void to_json(json& dst, const layer_desc_t& src) {
        ZoneScoped;

        dst["path"] = src.path;
        dst["size"] = src.size;

        std::string function_name;
        switch (src.function) {
        case activation_function::sigmoid:
            function_name = "sigmoid";
            break;
        default:
            throw std::runtime_error("invalid activation function!");
        }

        dst["function"] = function_name;
    }

    void from_json(const json& src, network_desc_t& dst) {
        ZoneScoped;

        src["input_count"].get_to(dst.input_count);
        src["layers"].get_to(dst.layers);
    }

    void to_json(json& dst, const network_desc_t& src) {
        ZoneScoped;

        dst["input_count"] = src.input_count;
        dst["layers"] = src.layers;
    }

    loader::loader(const fs::path& directory) {
        ZoneScoped;

        m_directory = directory;
        m_file = m_directory / "network.json";

        if (!fs::exists(m_directory)) {
            fs::create_directories(m_directory);
        } else if (fs::is_regular_file(m_directory)) {
            throw std::runtime_error("cannot use a file as a directory!");
        }
    }

    static constexpr std::endian serialization_endianness = std::endian::little;
    static number_t read_number(file_decompressor& file, std::vector<uint8_t>& buffer) {
        ZoneScoped;
        if (buffer.size() < sizeof(number_t)) {
            buffer.resize(sizeof(number_t));
        }

        number_t result;
        file.read(buffer.data(), sizeof(number_t));

        read_with_endianness<serialization_endianness>(buffer.data(), result);
        return result;
    }

    static void write_number(file_compressor& file, number_t value, std::vector<uint8_t>& buffer) {
        ZoneScoped;
        if (buffer.size() < sizeof(number_t)) {
            buffer.resize(sizeof(number_t));
        }

        write_with_endianness<serialization_endianness>(value, buffer.data());
        file.write(buffer.data(), sizeof(number_t));
    }

    bool loader::load_from_file() {
        ZoneScoped;
        if (m_network) {
            return false;
        }

        std::fstream desc_stream(m_file, std::ios::in);
        if (!desc_stream.is_open()) {
            return false;
        }

        json desc;
        desc_stream >> desc;
        desc_stream.close();

        network_desc_t network_desc;
        desc.get_to(network_desc);

        std::vector<layer_t> layers(network_desc.layers.size());
        std::vector<uint8_t> buffer;

        for (size_t i = 0; i < layers.size(); i++) {
            auto& layer = layers[i];
            const auto& layer_desc = network_desc.layers[i];

            layer.function = layer_desc.function;
            layer.size = layer_desc.size;
            layer.previous_size =
                i > 0 ? network_desc.layers[i - 1].size : network_desc.input_count;

            layer.biases.resize(layer.size);
            layer.weights.resize(layer.size * layer.previous_size);

            auto data_file_path = m_directory / layer_desc.path;
            if (!fs::is_regular_file(data_file_path)) {
                return false;
            }

            file_decompressor data_file(data_file_path);
            for (uint64_t b = 0; b < layer.size; b++) {
                layer.biases[b] = read_number(data_file, buffer);
            }

            for (uint64_t w = 0; w < layer.size * layer.previous_size; w++) {
                layer.weights[w] = read_number(data_file, buffer);
            }
        }

        m_network = unique(new network(layers));
        return true;
    }

    bool loader::save_to_file() {
        ZoneScoped;
        if (!m_network) {
            return false;
        }

        const auto& layers = m_network->get_layers();
        if (layers.empty()) {
            return false;
        }

        network_desc_t desc;
        desc.input_count = layers[0].previous_size;
        desc.layers.resize(layers.size());

        std::vector<uint8_t> buffer;
        for (size_t i = 0; i < layers.size(); i++) {
            const auto& layer = layers[i];
            auto& layer_descs = desc.layers[i];

            layer_descs.function = layer.function;
            layer_descs.size = layer.size;
            layer_descs.path = std::to_string(i) + ".dat";

            file_compressor data_file(m_directory / layer_descs.path);
            for (size_t b = 0; b < layer.biases.size(); b++) {
                write_number(data_file, layer.biases[b], buffer);
            }

            for (size_t w = 0; w < layer.weights.size(); w++) {
                write_number(data_file, layer.weights[w], buffer);
            }
        }

        json desc_data = desc;
        std::fstream desc_file(m_file, std::ios::out);

        desc_file << desc_data.dump(4);
        desc_file.close();

        return true;
    }

    bool loader::load_from_memory(network* nn) {
        ZoneScoped;
        if (m_network) {
            return false;
        }

        m_network = unique(nn);
        return true;
    }
} // namespace neuralnet