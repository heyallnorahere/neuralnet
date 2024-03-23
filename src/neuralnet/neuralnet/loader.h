#pragma once
#include "neuralnet/network.h"

namespace neuralnet {
    class loader {
    public:
        loader(const fs::path& directory);
        ~loader() = default;

        loader(const loader&) = delete;
        loader& operator=(const loader&) = delete;

        // will load and allocate a new network from file
        bool load_from_file();

        // will save the loaded network to disk, if one exists
        bool save_to_file();

        // will take ownership of network. use with caution
        bool load_from_memory(network* nn);

        // checks if the loader has a network loaded
        bool has_network_loaded() const { return m_network.get() != nullptr; }

        // releases the currently loaded network from this loader. if no network is present, returns
        // nullptr
        network* release_network() { return m_network.release(); }

    private:
        fs::path m_directory, m_file;
        std::unique_ptr<network> m_network;
    };
} // namespace neuralnet