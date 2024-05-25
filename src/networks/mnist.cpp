#include <memory>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <limits>

#include <neuralnet.h>
#include <neuralnet/compression.h>
#include <neuralnet/resources.h>
using number_t = neuralnet::number_t;

struct group_paths_t {
    neuralnet::fs::path images, labels;
};

struct mnist_sample_t {
    std::vector<number_t> image, label;
};

class mnist_dataset : public neuralnet::dataset {
public:
    static constexpr uint64_t output_count = 10;

    static uint32_t read_uint32_big_endian(neuralnet::file_decompressor& src, void* buffer) {
        ZoneScoped;
        if (src.read(buffer, sizeof(uint32_t)) < sizeof(uint32_t)) {
            return 0;
        }

        uint32_t result;
        neuralnet::read_with_endianness<std::endian::big>(buffer, result);

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

        neuralnet::file_decompressor images_file(paths.images);
        neuralnet::file_decompressor labels_file(paths.labels);
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

int main(int argc, const char** argv) {
    ZoneScoped;

    neuralnet::trainer_settings_t settings;
    settings.batch_size = 100;
    settings.eval_batch_size = 100;
    settings.learning_rate = 0.1;
    settings.minimum_average_cost = 1;

    auto evaluator = neuralnet::unique(neuralnet::evaluators::choose_evaluator());
    auto dataset = neuralnet::unique(new mnist_dataset);

    if (!evaluator) {
        std::cerr << "no evaluator available!" << std::endl;
        return 1;
    }

    std::unique_ptr<neuralnet::network> network;
    neuralnet::loader loader(neuralnet::fs::current_path() / "network");
    
    if (loader.load_from_file()) {
        network = neuralnet::unique(loader.release_network());
    } else {
        static const std::vector<uint64_t> layer_sizes = { dataset->get_input_count(), 128, 64,
                                                           dataset->get_output_count() };

        network = neuralnet::unique(
            neuralnet::network::randomize(layer_sizes, neuralnet::activation_function::sigmoid));

        loader.load_from_memory(network.get());
        loader.save_to_file();
        loader.release_network();
    }

    auto trainer = neuralnet::unique(
        new neuralnet::trainer(network.get(), evaluator.get(), dataset.get(), settings));

    trainer->start();
    while (trainer->is_running()) {
        trainer->update();
    }

    loader.load_from_memory(network.get());
    loader.save_to_file();
    loader.release_network();

    return 0;
}