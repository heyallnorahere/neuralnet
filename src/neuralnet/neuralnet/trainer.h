#pragma once
#include "neuralnet/network.h"
#include "neuralnet/evaluator.h"

namespace neuralnet {
    struct trainer_settings_t {
        uint64_t batch_size, eval_batch_size;
        number_t learning_rate;
        number_t minimum_average_cost;

        number_t (*cost)(number_t x, number_t y);
        number_t (*cost_derivative)(number_t x, number_t y);
    };

    enum class dataset_group { training, testing, evaluation };
    enum class training_stage { eval, backprop, deltas };

    class NN_API dataset {
    public:
        virtual ~dataset() = default;

        virtual uint64_t get_input_count() const = 0;
        virtual uint64_t get_output_count() const = 0;

        virtual void get_groups(std::unordered_set<dataset_group>& groups) const = 0;
        virtual uint64_t get_sample_count(dataset_group group) const = 0;

        virtual bool get_sample(dataset_group group, uint64_t sample, std::vector<number_t>& inputs,
                                std::vector<number_t>& outputs) const = 0;
    };

    class NN_API trainer {
    public:
        trainer(network* nn, evaluator* nn_evaluator, dataset* data,
                const trainer_settings_t& settings);
        ~trainer();

        trainer(const trainer&) = delete;
        trainer& operator=(const trainer&) = delete;

        trainer_settings_t& get_settings() { return m_settings; }
        const trainer_settings_t& get_settings() const { return m_settings; }

        bool is_running() { return m_running; }

        std::optional<number_t> compute_test_cost();

        void start();
        void stop();
        void update();

    private:
        struct sample_id_t {
            uint64_t sample;
            dataset_group group;
        };

        void regenerate_training_cycle();

        void eval();
        void backprop();
        bool compose_deltas();
        bool do_training_cycle();

        bool check_eval_keys();
        bool do_eval();

        network* m_network;
        evaluator* m_evaluator;
        dataset* m_dataset;
        trainer_settings_t m_settings;

        trainer_settings_t m_current_settings;
        uint64_t m_batch_count, m_current_batch, m_current_eval_index;
        bool m_running;
        std::vector<number_t> m_eval_costs;
        std::unordered_map<uint64_t, std::vector<number_t>> m_sample_map;
        std::vector<uint64_t> m_training_cycle;

        dataset_group m_phase;
        training_stage m_stage;
        std::vector<uint64_t> m_current_eval_keys;
    };
} // namespace neuralnet