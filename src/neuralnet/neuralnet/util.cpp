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

    namespace random {
        class generator {
        public:
            generator() {
                m_rng.seed(m_device());
            }

            generator(const generator&) = delete;
            generator& operator=(const generator&) = delete;

            std::mt19937_64& get_rng() { return m_rng; }

        private:
            std::random_device m_device;
            std::mt19937_64 m_rng;
        };

        static generator s_rng;
        std::mt19937_64& rng() { return s_rng.get_rng(); }
    } // namespace random
} // namespace neuralnet

void* operator new(size_t size) { return neuralnet::alloc(size); }
void operator delete(void* block) { return neuralnet::freemem(block); }