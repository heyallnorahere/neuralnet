#include "nnpch.h"
#include "neuralnet/trainer.h"
#include "neuralnet/util.h"

namespace neuralnet {
    trainer::trainer(network* nn, evaluator* nn_evaluator, dataset* data,
                     const trainer_settings_t& settings) {
        ZoneScoped;

        m_network = nn;
        m_evaluator = nn_evaluator;
        m_dataset = data;

        m_settings = settings;
        m_running = false;
    }

    trainer::~trainer() {
        ZoneScoped;

        if (m_running) {
            stop();
        }
    }

    std::optional<number_t> trainer::compute_test_cost() {
        ZoneScoped;
        if (m_eval_costs.empty()) {
            return {};
        }

        const auto& layers = m_network->get_layers();
        const auto& last_layer = layers[layers.size() - 1];

        number_t average = 0;
        for (number_t cost : m_eval_costs) {
            average += std::abs(cost);
        }

        return average / m_eval_costs.size();
    }

    static bool dataset_has_group(const dataset* set, dataset_group group) {
        std::unordered_set<dataset_group> groups;
        set->get_groups(groups);

        return groups.find(group) != groups.end();
    }

    void trainer::start() {
        ZoneScoped;

        if (m_running) {
            return;
        }

        std::unordered_set<dataset_group> groups;
        m_dataset->get_groups(groups);

        if (!dataset_has_group(m_dataset, dataset_group::training)) {
            throw std::runtime_error("dataset has no training group!");
        }

        if (!dataset_has_group(m_dataset, dataset_group::testing)) {
            throw std::runtime_error("dataset has no testing group!");
        }

        m_phase = dataset_group::training;
        m_stage = training_stage::eval;
        m_current_settings = m_settings;

        uint64_t training_sample_count = m_dataset->get_sample_count(dataset_group::training);
        m_batch_count = (uint64_t)std::floor((long double)training_sample_count /
                                             (long double)m_current_settings.batch_size);

        m_running = true;
        regenerate_training_cycle();

        std::cout << "beginning training!" << std::endl;
    }

    void trainer::stop() {
        ZoneScoped;

        if (!m_running) {
            return;
        }

        std::cout << "stopping training" << std::endl;
        m_running = false; // lol
    }

    void trainer::update() {
        ZoneScoped;

        switch (m_phase) {
        case dataset_group::training:
            if (do_training_cycle()) {
                m_phase = dataset_group::testing;
                m_current_eval_index = 0;
            }

            break;
        default:
            if (do_eval()) {
                auto cost = compute_test_cost();
                if (cost) {
                    number_t cost_value = cost.value();
                    std::cout << "cost: " << cost_value << std::endl;

                    if (cost_value < m_current_settings.minimum_average_cost) {
                        switch (m_phase) {
                        case dataset_group::testing:
                            if (dataset_has_group(m_dataset, dataset_group::evaluation)) {
                                m_phase = dataset_group::evaluation;
                                m_current_eval_index = 0;
                            } else {
                                stop();
                            }

                            break;
                        case dataset_group::evaluation:
                            stop();
                            break;
                        }
                    } else {
                        m_phase = dataset_group::training;
                    }
                }
            }

            break;
        }
    }

    void trainer::regenerate_training_cycle() {
        ZoneScoped;
        m_current_batch = 0;

        m_training_cycle.resize((size_t)m_dataset->get_sample_count(dataset_group::training));
        for (size_t i = 0; i < m_training_cycle.size(); i++) {
            m_training_cycle[i] = i;
        }

        size_t n = m_training_cycle.size() - 1;
        while (n > 1) {
            size_t i = random::next<size_t>(0, n--);
            std::swap(m_training_cycle[i], m_training_cycle[n]);
        }
    }

    void trainer::eval() {
        ZoneScoped;

        uint64_t batch_size = m_current_settings.batch_size;
        for (uint64_t i = 0; i < batch_size; i++) {
            uint64_t training_cycle_index = i + m_current_batch * batch_size;
            uint64_t sample_index = m_training_cycle[(size_t)training_cycle_index];

            std::vector<number_t> inputs, outputs;
            if (!m_dataset->get_sample(dataset_group::training, sample_index, inputs, outputs)) {
                throw std::runtime_error("failed to retrieve sample " +
                                         std::to_string(sample_index) + "!");
            }

            auto key = m_evaluator->begin_eval(m_network, inputs);
            if (!key) {
                throw std::runtime_error("failed to begin evaluation!");
            }

            uint64_t eval_key = key.value();
            m_sample_map[eval_key] = outputs;
            m_current_eval_keys.push_back(eval_key);
        }

        m_evaluator->flush();
    }

    void trainer::backprop() {
        ZoneScoped;

        if (m_current_eval_keys.empty()) {
            return;
        }

        std::vector eval_keys(m_current_eval_keys);
        m_current_eval_keys.clear();

        for (uint64_t eval_key : eval_keys) {
            if (m_sample_map.find(eval_key) == m_sample_map.end()) {
                throw std::runtime_error("failed to find sample expected outputs!");
            }

            backprop_data_t data;
            data.expected_outputs = m_sample_map[eval_key];

            if (!m_evaluator->get_eval_result(eval_key, &data.eval_outputs)) {
                throw std::runtime_error("failed to retrieve eval result!");
            }

            auto key = m_evaluator->begin_backprop(m_network, data);
            if (!key) {
                throw std::runtime_error("failed to begin backpropagation!");
            }

            m_sample_map.erase(eval_key);
            m_evaluator->free_result(eval_key);
            m_current_eval_keys.push_back(key.value());
        }

        m_evaluator->flush();
    }

    bool trainer::compose_deltas() {
        ZoneScoped;

        auto& layers = m_network->get_layers();
        size_t layer_count = layers.size();

        number_t delta_scalar = m_current_settings.learning_rate / m_current_settings.batch_size;
        for (uint64_t backprop_key : m_current_eval_keys) {
            std::vector<layer_t> deltas;
            if (!m_evaluator->get_backprop_result(backprop_key, deltas)) {
                throw std::runtime_error("failed to retrieve layer deltas!");
            }

            if (layer_count != deltas.size()) {
                throw std::runtime_error("delta/layer count mismatch!");
            }

            for (size_t i = 0; i < layer_count; i++) {
                auto& layer = layers[i];
                auto& delta = deltas[i];

                if (delta.size != layer.size || delta.previous_size != layer.previous_size) {
                    throw std::runtime_error("delta/layer size mismatch!");
                }

                for (size_t c = 0; c < layer.size; c++) {
                    number_t& bias = network::get_bias_address(layer, c);
                    number_t bias_delta = network::get_bias(delta, c);
                    bias -= bias_delta * delta_scalar;

                    for (size_t p = 0; p < layer.previous_size; p++) {
                        number_t& weight = network::get_weight_address(layer, c, p);
                        number_t weight_delta = network::get_weight(delta, c, p);
                        weight -= weight_delta * delta_scalar;
                    }
                }
            }

            m_evaluator->free_result(backprop_key);
        }

        m_current_eval_keys.clear();
        std::cout << "finished batch " << m_current_batch << std::endl;

        return ++m_current_batch == m_batch_count;
    }

    bool trainer::do_training_cycle() {
        ZoneScoped;

        while (true) {
            bool should_wait = false;
            for (uint64_t key : m_current_eval_keys) {
                should_wait |= !m_evaluator->is_result_ready(key);
            }

            if (should_wait) {
                return false;
            } else if (!m_current_eval_keys.empty()) {
                switch (m_stage) {
                case training_stage::eval:
                    m_stage = training_stage::backprop;
                    break;
                case training_stage::backprop:
                    m_stage = training_stage::deltas;
                    break;
                }

                // delta composition is always cpu-bound
            }

            switch (m_stage) {
            case training_stage::eval:
                eval();
                break;
            case training_stage::backprop:
                backprop();
                break;
            case training_stage::deltas:
                m_stage = training_stage::eval;
                if (compose_deltas()) {
                    regenerate_training_cycle();
                    return true;
                }

                return false;
            }
        }
    }

    struct output_pair_t {
        std::vector<number_t> outputs, expected_outputs;
    };

    bool trainer::check_eval_keys() {
        ZoneScoped;

        std::vector<number_t> costs;
        for (uint64_t key : m_current_eval_keys) {
            void* output;
            if (!m_evaluator->get_eval_result(key, &output)) {
                return true;
            }

            if (m_sample_map.find(key) == m_sample_map.end()) {
                throw std::runtime_error("failed to find expected outputs for a sample!");
            }

            std::vector<number_t> outputs;
            m_evaluator->retrieve_eval_values(m_network, output, outputs);

            const auto& expected_outputs = m_sample_map[key];
            for (size_t i = 0; i < outputs.size(); i++) {
                number_t cost = m_evaluator->cost_function(outputs[i], expected_outputs[i]);
                costs.push_back(cost);
            }

            m_sample_map.erase(key);
        }

        m_eval_costs.insert(m_eval_costs.end(), costs.begin(), costs.end());
        return false;
    }

    bool trainer::do_eval() {
        ZoneScoped;

        if (!m_current_eval_keys.empty()) {
            if (check_eval_keys()) {
                return false;
            }

            m_current_eval_index += m_current_eval_keys.size();
        }

        uint64_t sample_count = m_dataset->get_sample_count(m_phase);
        uint64_t batch_size =
            std::min(sample_count - m_current_eval_index, m_current_settings.eval_batch_size);

        if (batch_size == 0) {
            m_current_eval_keys.clear();
            return true;
        }

        m_current_eval_keys.resize(batch_size);
        for (uint64_t i = 0; i < batch_size; i++) {
            uint64_t sample = i + m_current_eval_index;

            std::vector<number_t> inputs, outputs;
            if (!m_dataset->get_sample(m_phase, sample, inputs, outputs)) {
                throw std::runtime_error("failed to get eval sample!");
            }

            auto key = m_evaluator->begin_eval(m_network, inputs);
            if (!key) {
                throw std::runtime_error("failed to begin eval!");
            }

            uint64_t eval_key = key.value();
            m_sample_map[eval_key] = outputs;
        }

        m_evaluator->flush();
        if (check_eval_keys()) {
            return false;
        }

        m_current_eval_index += m_current_eval_keys.size();
        if (m_current_eval_index == sample_count) {
            m_current_eval_keys.clear();
            return true;
        }

        return false;
    }
} // namespace neuralnet