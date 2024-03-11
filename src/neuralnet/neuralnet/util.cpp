#include "nnpch.h"
#include "neuralnet/util.h"

namespace neuralnet {
    void* alloc(size_t size) {
        void* ptr = std::malloc(size);
        TracyAlloc(ptr, size);

        return ptr;
    }

    void freemem(void* block) {
        std::free(block);
        TracyFree(block);
    }

    void copy(const void* src, void* dst, size_t size) { std::memcpy(dst, src, size); }
} // namespace neuralnet

void* operator new(size_t size) { return neuralnet::alloc(size); }
void operator delete(void* block) { return neuralnet::freemem(block); }