#include "nnpch.h"
#include "neuralnet/compression.h"

#include <zlib.h>

namespace neuralnet {
    file_decompressor::file_decompressor(const neuralnet::fs::path& path) {
        ZoneScoped;
        m_file = gzopen(path.string().c_str(), "rb");
    }

    file_decompressor::~file_decompressor() {
        ZoneScoped;
        gzclose(m_file);
    }

    size_t file_decompressor::get_position() const {
        ZoneScoped;
        return gztell(m_file);
    }

    int32_t file_decompressor::read(void* buffer, uint32_t buffer_size) {
        ZoneScoped;
        return (int32_t)gzread(m_file, buffer, (unsigned)buffer_size);
    }

    file_compressor::file_compressor(const neuralnet::fs::path& path) {
        ZoneScoped;
        m_file = gzopen(path.string().c_str(), "wb");
    }

    file_compressor::~file_compressor() {
        ZoneScoped;
        gzclose(m_file);
    }

    size_t file_compressor::get_position() const {
        ZoneScoped;
        return gztell(m_file);
    }

    int32_t file_compressor::write(const void* buffer, uint32_t buffer_size) {
        ZoneScoped;
        return (int32_t)gzwrite(m_file, buffer, (unsigned)buffer_size);
    }
} // namespace neuralnet