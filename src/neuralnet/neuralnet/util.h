#pragma once

void* operator new(size_t size);
void operator delete(void* block);

namespace neuralnet {
    template <typename _Ty>
    struct nn_delete {
        inline void operator()(_Ty* block) const { delete block; }
    };

    template <typename _Ty>
    using unique_ptr = std::unique_ptr<_Ty, nn_delete<_Ty>>;

    template <typename _Ty>
    inline unique_ptr<_Ty> unique(_Ty* block) {
        return unique_ptr<_Ty>(block);
    }

    void* alloc(size_t size);
    void freemem(void* block);
    void copy(const void* src, void* dst, size_t size);
} // namespace neuralnet