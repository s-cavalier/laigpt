#ifndef __POLY_REF_HPP__
#define __POLY_REF_HPP__

#include <cstddef>
#include <concepts>
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

template <class Base>
concept SBOQualified = requires(const Base a) {
    a.stack_clone( (void*)nullptr );
    { a.heap_clone() } -> std::same_as<Base*>;
};

template <class Base, size_t SBO_SIZE = 32> requires (SBOQualified<Base>)
class PolyRef {
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
        case SBO:  return reinterpret_cast<Base*>(_storage.sbo);
        default:   return nullptr;
        }
    }

    const Base* raw_ptr() const {
        switch (_mode) {
        case Static:
        case Heap: return _storage.ptr;
        case SBO:  return reinterpret_cast<const Base*>(_storage.sbo);
        default:   return nullptr;
        }
    }

    void destroy() {
        switch (_mode) {
        case Static:
            break;
        case SBO:
            if (auto* p = raw_ptr()) p->~Base();
            break;
        case Heap:
            delete _storage.ptr;
            _storage.ptr = nullptr;
            break;
        default:
            break;
        }
        _mode = Unspecified;
    }

    void move_from(PolyRef&& other) {
        _mode = other._mode;

        switch (other._mode) {
        case Static:
            _storage.ptr = other._storage.ptr;
            break;

        case SBO:
            std::memcpy(_storage.sbo, other._storage.sbo, SBO_SIZE);
            if (auto* p = other.raw_ptr()) p->~Base();
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

public:
    PolyRef() = default;

    PolyRef(const PolyRef& other) {
        _mode = other._mode;

        switch (_mode) {
        case Static:
            _storage.ptr = other._storage.ptr;
            break;

        case SBO:
            other.raw_ptr()->stack_clone(_storage.sbo);
            break;

        case Heap:
            _storage.ptr = other.raw_ptr()->heap_clone();
            break;

        default:
            _storage.ptr = nullptr;
            break;
        }
    }

    PolyRef& operator=(const PolyRef& other) {
        if (this == &other) return *this;

        destroy();
        _mode = other._mode;

        switch (_mode) {
        case Static:
            _storage.ptr = other._storage.ptr;
            break;

        case SBO:
            other.raw_ptr()->stack_clone(_storage.sbo);
            break;

        case Heap:
            _storage.ptr = other.raw_ptr()->heap_clone();
            break;

        default:
            _storage.ptr = nullptr;
            break;
        }

        return *this;
    }

    PolyRef(PolyRef&& other) noexcept {
        move_from(std::move(other));
    }

    PolyRef& operator=(PolyRef&& other) noexcept {
        if (this != &other) {
            destroy();
            move_from(std::move(other));
        }
        return *this;
    }

    ~PolyRef() {
        destroy();
    }


    template<class T, class... Args> requires ( std::derived_from<T, Base> && std::constructible_from<T, Args...> )
    static PolyRef emplace(Args&&... args) {

        PolyRef p;

        constexpr bool is_static = sizeof(T) == sizeof(void*); 
        // If it just has one vptr it'll be size at least 8, and if it onyl has the vptr, it's size 8. It must have a vptr since we statically assert it derives from Base.

        constexpr bool fits_sbo = sizeof(T) <= SBO_SIZE && alignof(T) <= alignof(std::max_align_t);

        if constexpr (is_static) {
            static T singleton{};
            p._mode = Static;
            p._storage.ptr = &singleton;
        }

        else if constexpr (fits_sbo) {
            void* buf = p._storage.sbo;
            new (buf) T(std::forward<Args>(args)...);
            p._mode = SBO;
        }

        else {
            p._storage.ptr = new T(std::forward<Args>(args)...);
            p._mode = Heap;
        }

        return p;
    }

    Base*       get()       { return raw_ptr(); }
    const Base* get() const { return raw_ptr(); }

    Base&       operator*()       { return *raw_ptr(); }
    const Base& operator*() const { return *raw_ptr(); }

    Base*       operator->()       { return raw_ptr(); }
    const Base* operator->() const { return raw_ptr(); }

    explicit operator bool() const { return _mode != Unspecified; }

    Mode mode() const { return _mode; }
};



#endif