#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <cstddef>
#include <memory>

namespace ag {

    class TensorNetwork {

        class MemoryArena {
            std::byte* buffer;
            size_t size;
            size_t offset;

        public:
            explicit MemoryArena(size_t bytes);
            ~MemoryArena() noexcept;

            MemoryArena(const MemoryArena&) = delete;
            MemoryArena& operator=(const MemoryArena&) = delete;

            MemoryArena(MemoryArena&& other) = delete;
            MemoryArena& operator=(MemoryArena&& other) = delete;

            template <class T>
            T* allocate(size_t n = 1) {
                constexpr size_t alignment = alignof(T);
                constexpr size_t bytes = n * sizeof(T);

                void* ptr = (void*)(buffer + offset);
                size_t space = size - offset;
                
                if ( !std::align( alignment, bytes, ptr, space ) ) throw std::bad_alloc("TensorNetwork ran out of memory. Try using an obj with more memory.");
                
                offset = ( (std::byte*)(ptr) - buffer ) + bytes;
                return (T*)ptr;
            }

            void reset() noexcept { offset = 0; }


        };






    };



}


#endif
