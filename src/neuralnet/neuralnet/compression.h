#pragma once

struct gzFile_s;
namespace neuralnet {
    class file_decompressor {
    public:
        file_decompressor(const neuralnet::fs::path& path);
        ~file_decompressor();

        file_decompressor(const file_decompressor&) = delete;
        file_decompressor& operator=(const file_decompressor&) = delete;

        off64_t get_position() const;
        int32_t read(void* buffer, uint32_t buffer_size);

    private:
        gzFile_s* m_file;
    };

    class file_compressor {
    public:
        file_compressor(const neuralnet::fs::path& path);
        ~file_compressor();

        file_compressor(const file_compressor&) = delete;
        file_compressor& operator=(const file_compressor&) = delete;

        off64_t get_position() const;
        int32_t write(const void* buffer, uint32_t buffer_size);

    private:
        gzFile_s* m_file;
    };
} // namespace neuralnet