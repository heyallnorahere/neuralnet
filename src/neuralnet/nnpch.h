#pragma once
#define NN_PCH_INCLUDED

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <optional>

#include <stddef.h>

namespace neuralnet {
    using number_t = double;

    inline void* alloc(size_t size) { return std::malloc(size); }
    inline void freemem(void* block) { std::free(block); }
    inline void copy(const void* src, void* dst, size_t size) { std::memcpy(dst, src, size); }
} // namespace neuralnet

// export macro
#if defined(_WIN32) && defined(NN_SHARED)
#ifdef NN_EXPORT
#define NN_API __declspec(dllexport)
#else
#define NN_API __declspec(dllimport)
#endif
#else
#define NN_API
#endif