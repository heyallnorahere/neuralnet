#include <memory>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <bit>
#include <limits>

#include <neuralnet.h>
#include <zlib.h>
using number_t = neuralnet::number_t;

class decompressor {
public:
    decompressor(const neuralnet::fs::path& path) {
        ZoneScoped;
        m_file = gzopen64(path.string().c_str(), "rb");
    }

    ~decompressor() {
        ZoneScoped;
        gzclose(m_file);
    }

    off64_t get_position() const {
        ZoneScoped;
        return gztell64(m_file);
    }

    int32_t read(void* buffer, uint32_t buffer_size) {
        ZoneScoped;
        return (int32_t)gzread(m_file, buffer, (unsigned)buffer_size);
    }

    decompressor(const decompressor&) = delete;
    decompressor& operator=(const decompressor&) = delete;

private:
    gzFile m_file;
};

template <std::endian E, typename T>
static void read_with_endianness(const void* data, T& result) {
    ZoneScoped;
    if constexpr (std::endian::native == E) {
        neuralnet::copy(data, &result, sizeof(T));
    } else {
        const uint8_t* first = (const uint8_t*)(const void*)data;
        uint8_t* result_last = (uint8_t*)((size_t)&result + sizeof(T));
        
        for (size_t i = 0; i < sizeof(T); i++) {
            *(--result_last) = *(first++);
        }
    }
}

struct group_paths_t {
    neuralnet::fs::path images, labels;
};

struct mnist_sample_t {
    std::vector<number_t> image, label;
};

class mnist_dataset : public neuralnet::dataset {
public:
    static constexpr uint64_t output_count = 10;

    static uint32_t read_uint32_big_endian(decompressor& src, void* buffer) {
        ZoneScoped;
        if (src.read(buffer, sizeof(uint32_t)) < sizeof(uint32_t)) {
            return 0;
        }

        uint32_t result;
        read_with_endianness<std::endian::big>(buffer, result);

        return result;
    }

    mnist_dataset() {
        ZoneScoped;
        static const std::unordered_map<neuralnet::dataset_group, group_paths_t> paths = {
            { neuralnet::dataset_group::training,
              { "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz" } },
            { neuralnet::dataset_group::testing,
              { "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz" } }
        };

        m_input_count = 0;
        for (const auto& [group, group_paths] : paths) {
            load_mnist_group(group_paths, m_groups[group]);
        }

        if (m_input_count == 0) {
            throw std::runtime_error("invalid data!");
        }
    }

    virtual ~mnist_dataset() override = default;

    mnist_dataset(const mnist_dataset&) = delete;
    mnist_dataset& operator=(const mnist_dataset&) = delete;

    virtual uint64_t get_input_count() const override { return m_input_count; }
    virtual uint64_t get_output_count() const override { return output_count; }

    virtual void get_groups(std::unordered_set<neuralnet::dataset_group>& groups) const override {
        ZoneScoped;

        groups.clear();
        for (const auto& [group, data] : m_groups) {
            groups.insert(group);
        }
    }

    virtual uint64_t get_sample_count(neuralnet::dataset_group group) const override {
        ZoneScoped;
        if (m_groups.find(group) == m_groups.end()) {
            return 0;
        }

        return m_groups.at(group).size();
    }

    virtual bool get_sample(neuralnet::dataset_group group, uint64_t sample,
                            std::vector<number_t>& inputs,
                            std::vector<number_t>& outputs) const override {
        ZoneScoped;
        if (m_groups.find(group) == m_groups.end()) {
            return false;
        }

        const auto& group_data = m_groups.at(group);
        if (sample >= group_data.size()) {
            return false;
        }

        const auto& sample_data = group_data[sample];
        inputs = sample_data.image;
        outputs = sample_data.label;

        return true;
    }

private:
    void load_mnist_group(const group_paths_t& paths, std::vector<mnist_sample_t>& samples) {
        ZoneScoped;

        decompressor images_file(paths.images);
        decompressor labels_file(paths.labels);
        uint8_t int_buffer[sizeof(uint32_t)];

        // see mnist manual
        if (read_uint32_big_endian(images_file, int_buffer) != 0x803) {
            throw std::runtime_error("invalid image magic number!");
        }

        // again, see mnist manual
        if (read_uint32_big_endian(labels_file, int_buffer) != 0x801) {
            throw std::runtime_error("invalid label magic number!");
        }

        uint32_t sample_count = read_uint32_big_endian(images_file, int_buffer);
        uint32_t row_count = read_uint32_big_endian(images_file, int_buffer);
        uint32_t column_count = read_uint32_big_endian(images_file, int_buffer);

        if (sample_count != read_uint32_big_endian(labels_file, int_buffer)) {
            throw std::runtime_error("sample count mismatch!");
        }

        if (m_input_count == 0) {
            m_input_count = (uint64_t)row_count * column_count;
        } else if (m_input_count != (uint64_t)row_count * column_count) {
            throw std::runtime_error("input count mismatch!");
        }

        std::vector<uint8_t> image_data(sample_count * row_count * column_count);
        std::vector<uint8_t> label_data(sample_count);

        size_t image_bytes_read = 0;
        while (true) {
            int32_t bytes_read = images_file.read(&image_data[image_bytes_read],
                                                  (uint32_t)(image_data.size() - image_bytes_read));
            if (bytes_read <= 0) {
                break;
            }

            image_bytes_read += bytes_read;
        }

        size_t label_bytes_read = 0;
        while (true) {
            int32_t bytes_read = labels_file.read(&label_data[label_bytes_read],
                                                  (uint32_t)(label_data.size() - label_bytes_read));
            if (bytes_read <= 0) {
                break;
            }

            label_bytes_read += bytes_read;
        }

        samples.clear();
        for (size_t i = 0; i < sample_count; i++) {
            auto& sample = samples.emplace_back();
            sample.image.resize(m_input_count);

            size_t image_sample_offset = i * m_input_count;
            for (size_t j = 0; j < m_input_count; j++) {
                uint8_t byte = image_data[image_sample_offset + j];
                sample.image[j] = (number_t)byte / std::numeric_limits<uint8_t>::max();
            }

            uint8_t label = label_data[i];
            for (uint64_t j = 0; j < output_count; j++) {
                sample.label.push_back(label == j ? 1 : 0);
            }
        }
    }

    std::unordered_map<neuralnet::dataset_group, std::vector<mnist_sample_t>> m_groups;
    uint64_t m_input_count;
};

static neuralnet::network* create_network(const std::vector<uint64_t>& layer_sizes,
                                          const neuralnet::activation_function_t& function) {
    ZoneScoped;

    static constexpr number_t min = -1;
    static constexpr number_t max = 1;

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
            neuralnet::network::get_bias_address(layer, c) = neuralnet::random::next(min, max);

            for (size_t p = 0; p < layer.previous_size; p++) {
                neuralnet::network::get_weight_address(layer, c, p) =
                    neuralnet::random::next<number_t>(min, max);
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

static number_t string_to_number(const std::string& string) {
    ZoneScoped;

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

static number_t sigmoid(number_t x) { return 1 / (1 + std::exp(-x)); }
static number_t dsigmoid_dx(number_t x) {
    number_t sig = sigmoid(x);
    return sig * (1 - sig);
}

static number_t C(number_t x, number_t y) { return std::pow(x - y, 2); }
static number_t dC_dx(number_t x, number_t y) { return 2 * (x - y); }

int main(int argc, const char** argv) {
    ZoneScoped;

    neuralnet::activation_function_t function;
    function.get = sigmoid;
    function.get_derivative = dsigmoid_dx;

    neuralnet::trainer_settings_t settings;
    settings.batch_size = 100;
    settings.cost = C;
    settings.cost_derivative = dC_dx;
    settings.eval_batch_size = 100;
    settings.learning_rate = 0.1;
    settings.minimum_average_cost = 0.02;

    auto evaluator = neuralnet::unique(neuralnet::create_cpu_evaluator());
    auto dataset = neuralnet::unique(new mnist_dataset);

    std::vector<uint64_t> layer_sizes = {
        dataset->get_input_count(),
        128,
        64,
        dataset->get_output_count()
    };

    auto network = neuralnet::unique(create_network(layer_sizes, function));
    auto trainer = neuralnet::unique(
        new neuralnet::trainer(network.get(), evaluator.get(), dataset.get(), settings));

    trainer->start();
    while (trainer->is_running()) {
        trainer->update();
    }

    return 0;
}