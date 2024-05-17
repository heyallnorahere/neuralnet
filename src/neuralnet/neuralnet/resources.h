#pragma once

namespace neuralnet {
    class resource {
    public:
        static const resource& get(const fs::path& path);

        resource(const fs::path& path, std::initializer_list<uint8_t>&& data) {
            m_data = std::move(data);
            m_path = path;
        }

        ~resource() = default;

        resource(const resource& other) = delete;
        resource& operator=(const resource& other) = delete;

        size_t size() const { return m_data.size(); }
        const uint8_t* data() const { return m_data.data(); }
        const fs::path& path() const { return m_path; }

        std::vector<uint8_t>::const_iterator begin() const { return m_data.begin(); }
        std::vector<uint8_t>::const_iterator end() const { return m_data.end(); }

    private:
        fs::path m_path;
        std::vector<uint8_t> m_data;
    };
}