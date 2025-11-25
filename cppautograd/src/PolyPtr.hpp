#ifndef __POLY_PTR_HPP__
#define __POLY_PTR_HPP__

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <cstring>

/*
Polymorphic Pointer class <br>
It's intended as a slightly more runtime optimized unique ptr with the cost of a bit more memory, it can do:
1. If a static reference exists, the pointer can store a pointer to that static variable without running delete on it. 
This can prevent heap-allocs in the case of empty types, like Functors or callables derived from Base.
2. If the derived type is small in size, it will store it inline via SBO. The size of the SBO buffer is templated.
This helps prevent unnecessary heap allocs.
3. If the derived type is too large for the SBO, it effectively just acts as a slightly fatter unique_ptr.


*/
template <class Base, size_t SBO_SIZE = 32>
class PolyPtr {
public:
    enum Mode : uint8_t { Unspecified, Static, SBO, Heap };

private:
    union Storage {
        std::byte sbo[SBO_SIZE];
        Base*     ptr;

        Storage() {}
        ~Storage() {}
    } _storage{};

    Mode _mode = Unspecified;

    Base* raw_ptr() {
        switch (_mode) {
        case Static:
        case Heap: return _storage.ptr;
        case SBO: return reinterpret_cast<Base*>(_storage.sbo);
        default: return nullptr;
        }
    }

    const Base* raw_ptr() const {
        switch (_mode) {
        case Static:
        case Heap: return _storage.ptr;
        case SBO: return reinterpret_cast<const Base*>(_storage.sbo);
        default: return nullptr;
        }
    }

    void destroy() {
        switch (_mode) {
        case Static:
            break;

        case SBO: {
            Base* p = raw_ptr();
            if (p) {
                p->~Base();
            }
            break;
        }

        case Heap:
            delete _storage.ptr;
            _storage.ptr = nullptr;
            break;

        default:
            break;
        }

        _mode = Unspecified;
    }

    void move_from(PolyPtr&& other) {
        _mode = other._mode;

        switch (other._mode) {
        case Static:
            _storage.ptr = other._storage.ptr;
            break;

        case SBO:
            std::memcpy(_storage.sbo, other._storage.sbo, SBO_SIZE);
            break;

        case Heap:
            _storage.ptr = other._storage.ptr;
            other._storage.ptr = nullptr;
            break;

        default:
            break;
        }

        other._mode = Unspecified;
    }

    PolyPtr() = default;

public:
    PolyPtr(const PolyPtr&) = delete;
    PolyPtr& operator=(const PolyPtr&) = delete;

    PolyPtr(PolyPtr&& other) noexcept { move_from(std::move(other)); }

    PolyPtr& operator=(PolyPtr&& other) noexcept {
        if (this != &other) {
            destroy();
            move_from(std::move(other));
        }
        return *this;
    }

    ~PolyPtr() { destroy(); }

    template<class T, class... Args>
    static PolyPtr emplace(Args&&... args) {
        static_assert(std::is_base_of_v<Base, T>, "PolyPtr::emplace<T>: T must derive from Base");

        PolyPtr p;

        constexpr bool is_empty = std::is_empty_v<T>;
        constexpr bool fits_sbo =
            sizeof(T) <= SBO_SIZE &&
            alignof(T) <= alignof(std::max_align_t) &&
            std::is_trivially_copyable_v<T>;

        if constexpr (is_empty) {
            static T singleton{};
            p._mode = Static;
            p._storage.ptr = &singleton;

        } else if constexpr (fits_sbo) {
            void* buf = static_cast<void*>(p._storage.sbo);
            T* obj = new (buf) T(std::forward<Args>(args)...);
            p._mode = SBO;
            (void)obj;
        } else {
            T* obj = new T(std::forward<Args>(args)...);
            p._mode = Heap;
            p._storage.ptr = obj;
        }

        return p;
    }

    Base* get()             { return raw_ptr(); }
    const Base* get() const { return raw_ptr(); }

    Base& operator*()       { return *raw_ptr(); }
    const Base& operator*() const { return *raw_ptr(); }

    Base* operator->()             { return raw_ptr(); }
    const Base* operator->() const { return raw_ptr(); }

    explicit operator bool() const { return _mode != Unspecified; }

    Mode mode() const { return _mode; }

    void reset() {
        destroy();
    }
};



#endif