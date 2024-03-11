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
#include <memory>

#include <stddef.h>

#include <tracy/Tracy.hpp>

namespace neuralnet {
    using number_t = double;
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