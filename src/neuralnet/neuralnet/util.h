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
    NN_API void copy(const void* src, void* dst, size_t size);

    template <std::endian E, typename T>
    inline void read_with_endianness(const void* data, T& result) {
        ZoneScoped;
        
        if constexpr (std::endian::native == E) {
            neuralnet::copy(data, &result, sizeof(T));
        } else {
            const uint8_t* first = (const uint8_t*)(const void*)data;
            uint8_t* result_last = (uint8_t*)((size_t)&result + sizeof(T));

            for (size_t i = 0; i < sizeof(T); i++) {
                *(--result_last) = *(first++);
            }
        }
    }

    template <std::endian E, typename T>
    inline void write_with_endianness(const T& data, void* result) {
        ZoneScoped;

        if constexpr (std::endian::native == E) {
            neuralnet::copy(&data, result, sizeof(T));
        } else {
            const uint8_t* first = (const uint8_t*)(const void*)&data;
            uint8_t* result_last = (uint8_t*)((size_t)result + sizeof(T));

            for (size_t i = 0; i < sizeof(T); i++) {
                *(--result_last) = *(first++);
            }
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