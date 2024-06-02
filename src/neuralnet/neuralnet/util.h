#pragma once

NN_API void* operator new(size_t size);
NN_API void operator delete(void* block);

namespace neuralnet {
    template <typename _Ty>
    inline std::unique_ptr<_Ty> unique(_Ty* block) {
        return std::unique_ptr<_Ty>(block);
    }

    NN_API void* alloc(size_t size);
    NN_API void freemem(void* block);
    NN_API void* reallocate(void* old_ptr, size_t new_size);
    NN_API void copy(const void* src, void* dst, size_t size);

    // https://stackoverflow.com/questions/38088732/explanation-to-aligned-malloc-implementation
    inline void* aligned_alloc(size_t size, size_t alignment) {
        void* p1;  // original block
        void** p2; // aligned block

        int offset = alignment - 1 + sizeof(void*);
        if ((p1 = (void*)alloc(size + offset)) == nullptr) {
            return nullptr;
        }

        p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
        p2[-1] = p1;
        return p2;
    }

    inline void aligned_free(void* block) {
        if (block == nullptr) {
            return;
        }

        freemem(((void**)block)[-1]);
    }

    template <std::endian E, typename T>
    inline void read_with_endianness(const void* data, T& result) {
        ZoneScoped;

        neuralnet::copy(data, &result, sizeof(T));
        if constexpr (std::endian::native != E) {
            std::reverse((uint8_t*)(void*)&result, (uint8_t*)((size_t)&result + sizeof(T)));
        }
    }

    template <std::endian E, typename T>
    inline void write_with_endianness(const T& data, void* result) {
        ZoneScoped;

        neuralnet::copy(&data, result, sizeof(T));
        if constexpr (std::endian::native != E) {
            std::reverse((uint8_t*)result, (uint8_t*)((size_t)result + sizeof(T)));
        }
    }

    namespace random {
        NN_API std::mt19937_64& rng();

        template <typename _Ty>
        inline std::enable_if_t<std::is_floating_point_v<_Ty>, _Ty> next(_Ty min, _Ty max) {
            std::uniform_real_distribution<_Ty> dist(min, max);
            return dist(rng());
        }

        template <typename _Ty>
        inline std::enable_if_t<std::is_integral_v<_Ty>, _Ty> next(_Ty min, _Ty max) {
            std::uniform_int_distribution<_Ty> dist(min, max);
            return dist(rng());
        }
    } // namespace random
} // namespace neuralnet