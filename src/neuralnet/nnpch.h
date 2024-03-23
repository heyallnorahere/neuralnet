#pragma once
#define NN_PCH_INCLUDED

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <memory>
#include <random>
#include <type_traits>
#include <iostream>
#include <bit>
#include <fstream>
#include <sstream>

#if __has_include(<filesystem>)
#include <filesystem>
#define NN_FS_INCLUDE_EXISTS
#else
#include <experimental/filesystem>
#endif

#include <stddef.h>

#include <tracy/Tracy.hpp>

namespace neuralnet {
    using number_t = double;

#ifdef NN_FS_INCLUDE_EXISTS
    namespace fs = std::filesystem;
#else
    namespace fs = std::experimental::filesystem;
#endif
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