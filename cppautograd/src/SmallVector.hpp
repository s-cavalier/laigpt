#ifndef __SMALL_VECTOR_HPP__
#define __SMALL_VECTOR_HPP__

#include <iterator>
#include <cstring>
#include <concepts>

/*
Small Vector implementation; uses SBO on a trivially copyable T. Optimal choices of SBO_BYTES are powers of 2 minus one, so 23, 31, 47, etc. based on 
how much space you want to alloc for SBO. Some convience choices are in the form of SVec{24, 32, 48}<T>
*/
template<class T, size_t SBO_BYTES> requires (std::is_trivially_copyable_v<T>)
class SmallVec {
    static_assert(SBO_BYTES >= sizeof(T), "SBO_BYTES must be >= sizeof(T).");

    struct HeapData {
        T*     ptr;     
        size_t size;    
        size_t cap;     
    };                  

    static constexpr size_t SBO_CAP = SBO_BYTES / sizeof(T);

    union Storage {
        HeapData heap;

        struct {
            alignas(T) std::byte data[SBO_BYTES];
            unsigned char size;       
        } sbo;

        Storage() {}
        ~Storage() {}
    } storage;

    bool using_sbo() const noexcept {
        return storage.sbo.size <= SBO_CAP;
    }

    bool using_heap() const noexcept {
        return storage.sbo.size > SBO_CAP;
    }

    void migrate_to_heap() {
        const T* arr = reinterpret_cast<const T*>(storage.sbo.data);

        size_t sz  = SBO_CAP;
        size_t cap = (SBO_CAP < 4 ? 4 : SBO_CAP * 2);

        T* newptr = static_cast<T*>(::operator new(cap * sizeof(T)));
        std::memcpy(newptr, arr, sz * sizeof(T));

        storage.heap.ptr  = newptr;
        storage.heap.size = sz;
        storage.heap.cap  = cap;

        storage.sbo.size = SBO_CAP + 1;
    }

    void grow_heap() {
        size_t newcap = storage.heap.cap * 2;
        T* newptr = static_cast<T*>(::operator new(newcap * sizeof(T)));

        std::memcpy(newptr, storage.heap.ptr, storage.heap.size * sizeof(T));

        ::operator delete(storage.heap.ptr);
        storage.heap.ptr = newptr;
        storage.heap.cap = newcap;
    }

public:

    SmallVec() noexcept {
        storage.sbo.size = 0;
    }

    SmallVec(std::initializer_list<T> init) {
        size_t n = init.size();

        if (n <= SBO_CAP) {
            storage.sbo.size = static_cast<unsigned char>(n);
            std::memcpy(storage.sbo.data, init.begin(), n * sizeof(T));
            return;
        }

        size_t cap = (n < 4 ? 4 : n);
        T* newptr = static_cast<T*>(::operator new(cap * sizeof(T)));

        std::memcpy(newptr, init.begin(), n * sizeof(T));

        storage.heap.ptr  = newptr;
        storage.heap.size = n;
        storage.heap.cap  = cap;

        storage.sbo.size = SBO_CAP + 1;
    }

    template <std::forward_iterator It>
    SmallVec( It from, It to ) {
        storage.sbo.size = 0;

        while ( from != to ) {
            emplace_back( *from );
            ++from;
        }
    }


    ~SmallVec() {
        if (using_heap()) ::operator delete(storage.heap.ptr);
    }

    SmallVec(const SmallVec& other) noexcept {
        if (other.using_sbo()) {
            storage = other.storage;
            return;
        }

        size_t sz  = other.storage.heap.size;
        size_t cap = other.storage.heap.cap;
        T* newptr = static_cast<T*>(::operator new(cap * sizeof(T)));

        std::memcpy(newptr, other.storage.heap.ptr, sz * sizeof(T));
        storage.heap.ptr  = newptr;
        storage.heap.size = sz;
        storage.heap.cap  = cap;

        storage.sbo.size = SBO_CAP + 1;
        
    }

    SmallVec& operator=(const SmallVec& other) noexcept {
        if (this == &other) return *this;
        this->~SmallVec();
        new (this) SmallVec(other);
        return *this;
    }

    SmallVec(SmallVec&& other) noexcept {
        if (other.using_sbo()) {
            storage = other.storage;
            return;
        }

        storage.heap = other.storage.heap;
        storage.sbo.size = SBO_CAP + 1;

        other.storage.sbo.size = 0;
    
    }

    SmallVec& operator=(SmallVec&& other) noexcept {
        if (this == &other) return *this;
        this->~SmallVec();
        new (this) SmallVec(std::move(other));
        return *this;
    }

    size_t size() const noexcept {
        return using_sbo() ? storage.sbo.size : storage.heap.size;
    }

    bool empty() const noexcept {
        return size() == 0;
    }

    const T& operator[](size_t i) const noexcept {
        return using_sbo() ? reinterpret_cast<const T*>(storage.sbo.data)[i] : storage.heap.ptr[i];
    }

    T& operator[](size_t i) noexcept {
        return using_sbo() ? reinterpret_cast<T*>(storage.sbo.data)[i] : storage.heap.ptr[i];
    }

    const T* data() const noexcept {
        return using_sbo() ? reinterpret_cast<const T*>(storage.sbo.data) : storage.heap.ptr;
    }

    T* data() noexcept {
        return using_sbo() ? reinterpret_cast<T*>(storage.sbo.data) : storage.heap.ptr;
    }

    void push_back(const T& v) {
        if (using_sbo()) {
            if (storage.sbo.size < SBO_CAP) {
                reinterpret_cast<T*>(storage.sbo.data)[storage.sbo.size++] = v;
                return;
            }
            migrate_to_heap();
        }

        if (storage.heap.size == storage.heap.cap) grow_heap();

        storage.heap.ptr[storage.heap.size++] = v;
    }

    void reserve(size_t new_cap) {
        if (using_sbo()) {
            if (new_cap <= SBO_CAP) return;

            size_t old_size = storage.sbo.size;
            const T* old_data = reinterpret_cast<const T*>(storage.sbo.data);

            std::size_t heap_cap = (new_cap < 4 ? 4 : new_cap);
            T* newptr = static_cast<T*>(::operator new(heap_cap * sizeof(T)));

            std::memcpy(newptr, old_data, old_size * sizeof(T));

            storage.heap.ptr  = newptr;
            storage.heap.size = old_size;
            storage.heap.cap  = heap_cap;

            storage.sbo.size = SBO_CAP + 1;
            return;
        }

        if (new_cap <= storage.heap.cap) return;

        size_t old_cap  = storage.heap.cap;
        size_t old_size = storage.heap.size;

        size_t heap_cap = new_cap;

        if (heap_cap < old_cap * 2) heap_cap = old_cap * 2;

        T* newptr = static_cast<T*>(::operator new(heap_cap * sizeof(T)));
        std::memcpy(newptr, storage.heap.ptr, old_size * sizeof(T));

        ::operator delete(storage.heap.ptr);
        storage.heap.ptr = newptr;
        storage.heap.cap = heap_cap;
    }


    void emplace_back(const T& v) {
        push_back(v);
    }

    template<typename... Args> requires (std::constructible_from<T, Args...>)
    void emplace_back(Args&&... args) {
        T temp{ std::forward<Args>(args)... };
        push_back(temp);
    }

    T& front() noexcept {
        return (*this)[0];
    }

    const T& front() const noexcept {
        return (*this)[0];
    }

    T& back() noexcept {
        return (*this)[size() - 1];
    }

    const T& back() const noexcept {
        return (*this)[size() - 1];
    }

    using iterator = T*;
    using const_iterator = const T*;

    iterator begin() noexcept {
        return data();
    }

    iterator end() noexcept {
        return data() + size();
    }

    const_iterator begin() const noexcept {
        return data();
    }

    const_iterator end() const noexcept {
        return data() + size();
    }

    const_iterator cbegin() const noexcept {
        return data();
    }

    const_iterator cend() const noexcept {
        return data() + size();
    }

};

template <class T>
using SVec24 = SmallVec<T, 23>;

template <class T>
using SVec32 = SmallVec<T, 31>;

template <class T>
using SVec48 = SmallVec<T, 47>;

template <class T>
using RefVec24 = SVec24< std::reference_wrapper< T >>;

template <class T>
using RefVec32 = SVec32< std::reference_wrapper< T >>;

#endif