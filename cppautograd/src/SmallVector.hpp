#ifndef __SMALL_VECTOR_HPP__
#define __SMALL_VECTOR_HPP__

#include <iterator>
#include <cstring>
#include <concepts>
#include <memory>

/*
Small Vector implementation; uses SBO on a trivially copyable T. Optimal choices of SBO_BYTES are powers of 2 minus one, so 23, 31, 47, etc. based on 
how much space you want to alloc for SBO. Some convience choices are in the form of SVec{24, 32, 48}<T>
*/
#include <memory>

template<class T, size_t SBO_BYTES, class Alloc = std::allocator<T>>
requires (std::is_trivially_copyable_v<T> && SBO_BYTES >= sizeof(T) )
class SmallVec {

    using AllocTraits = std::allocator_traits<Alloc>;

    Alloc alloc; // Can't be part of the union since we'd lose switching SBO -> heap.

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

    T* alloc_raw(size_t n) {
        return AllocTraits::allocate(alloc, n);
    }

    void free_raw(T* p, size_t n) {
        AllocTraits::deallocate(alloc, p, n);
    }

    void migrate_to_heap() {
        const T* arr = reinterpret_cast<const T*>(storage.sbo.data);

        size_t sz  = SBO_CAP;
        size_t cap = (SBO_CAP < 4 ? 4 : SBO_CAP * 2);

        T* newptr = alloc_raw(cap);
        std::memcpy(newptr, arr, sz * sizeof(T));

        storage.heap.ptr = newptr;
        storage.heap.size = sz;
        storage.heap.cap = cap;

        storage.sbo.size = SBO_CAP + 1;
    }

    void grow_heap() {
        size_t newcap = storage.heap.cap * 2;
        T* newptr = alloc_raw(newcap);

        std::memcpy(newptr, storage.heap.ptr, storage.heap.size * sizeof(T));

        free_raw(storage.heap.ptr, storage.heap.cap);
        storage.heap.ptr = newptr;
        storage.heap.cap = newcap;
    }

public:

    SmallVec(const Alloc& a = Alloc()) noexcept : alloc(a) {
        storage.sbo.size = 0;
    }

    SmallVec(std::initializer_list<T> init, const Alloc& a = Alloc()) : alloc(a) {
        size_t n = init.size();

        if (n <= SBO_CAP) {
            storage.sbo.size = static_cast<unsigned char>(n);
            std::memcpy(storage.sbo.data, init.begin(), n * sizeof(T));
            return;
        }

        size_t cap = (n < 4 ? 4 : n);
        T* newptr = alloc_raw(cap);

        std::memcpy(newptr, init.begin(), n * sizeof(T));

        storage.heap.ptr  = newptr;
        storage.heap.size = n;
        storage.heap.cap  = cap;

        storage.sbo.size = SBO_CAP + 1;
    }

    template <std::forward_iterator It>
    SmallVec(It from, It to, const Alloc& a = Alloc())
        : alloc(a)
    {
        storage.sbo.size = 0;
        while (from != to) {
            emplace_back(*from);
            ++from;
        }
    }

    ~SmallVec() {
        if (using_heap()) free_raw(storage.heap.ptr, storage.heap.cap);
    }

    SmallVec(const SmallVec& other) : alloc(AllocTraits::select_on_container_copy_construction(other.alloc)) {
        if (other.using_sbo()) {
            storage = other.storage;
            return;
        }

        size_t sz = other.storage.heap.size;
        size_t cap = other.storage.heap.cap;
        T* newptr = alloc_raw(cap);

        std::memcpy(newptr, other.storage.heap.ptr, sz * sizeof(T));
        storage.heap.ptr = newptr;
        storage.heap.size = sz;
        storage.heap.cap = cap;

        storage.sbo.size = SBO_CAP + 1;
    }

    SmallVec& operator=(const SmallVec& other) noexcept {
        if (this == &other) return *this;
        this->~SmallVec();
        new (this) SmallVec(other);
        return *this;
    }

    SmallVec(SmallVec&& other) noexcept : alloc(std::move(other.alloc)) {
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

    void reserve(size_t new_cap) {
        if (using_sbo()) {
            if (new_cap <= SBO_CAP) return;

            size_t old_size = storage.sbo.size;
            const T* old_data = reinterpret_cast<const T*>(storage.sbo.data);

            size_t heap_cap = (new_cap < 4 ? 4 : new_cap);
            T* newptr = alloc_raw(heap_cap);

            std::memcpy(newptr, old_data, old_size * sizeof(T));

            storage.heap.ptr = newptr;
            storage.heap.size = old_size;
            storage.heap.cap = heap_cap;
            storage.sbo.size = SBO_CAP + 1;
            return;
        }

        if (new_cap <= storage.heap.cap) return;

        size_t old_cap = storage.heap.cap;
        size_t old_size = storage.heap.size;

        size_t heap_cap = new_cap;
        if (heap_cap < old_cap * 2) heap_cap = old_cap * 2;

        T* newptr = alloc_raw(heap_cap);
        std::memcpy(newptr, storage.heap.ptr, old_size * sizeof(T));

        free_raw(storage.heap.ptr, old_cap);
        storage.heap.ptr = newptr;
        storage.heap.cap = heap_cap;
    }

    void change_allocator(const Alloc& new_alloc) {
        if (using_sbo()) {
            alloc = new_alloc;
            return;
        }

        size_t sz  = storage.heap.size;
        size_t cap = storage.heap.cap;

        Alloc old_alloc = std::move(alloc);
        alloc = new_alloc;

        T* newptr = AllocTraits::allocate(alloc, cap);
        std::memcpy(newptr, storage.heap.ptr, sz * sizeof(T));

        AllocTraits::deallocate(old_alloc, storage.heap.ptr, cap);

        storage.heap.ptr = newptr;
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

    void emplace_back(const T& v) {
        push_back(v);
    }

    template<typename... Args>
    void emplace_back(Args&&... args) {
        static_assert(std::is_trivially_copy_constructible_v<T>, "SmallVec<T>::emplace_back requires trivial T.");
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

    void resize(size_t new_size) {
        resize(new_size, T{});
    }

    void resize(size_t new_size, const T& value) {
        size_t cur = size();

        if (new_size <= cur) {
            if (using_sbo()) storage.sbo.size = static_cast<unsigned char>(new_size);

            else storage.heap.size = new_size;
            
            return;
        }

        size_t add = new_size - cur;

        if (using_sbo()) {
            if (new_size <= SBO_CAP) {
                T* arr = reinterpret_cast<T*>(storage.sbo.data);
                for (size_t i = 0; i < add; ++i) arr[cur + i] = value;

                storage.sbo.size = static_cast<unsigned char>(new_size);
                return;
            }

            migrate_to_heap();
            cur = storage.heap.size;
        }

        if (new_size > storage.heap.cap) reserve(new_size);
        
        for (size_t i = 0; i < add; ++i) storage.heap.ptr[cur + i] = value;

        storage.heap.size = new_size;
    }


};

// Same size as default vector.
template <class T, class Allocator = std::allocator<T> >
using SVec24 = SmallVec<T, 23, Allocator>;

// 8 extra bytes for SBO.
template <class T, class Allocator = std::allocator<T>>
using SVec32 = SmallVec<T, 31, Allocator>;

template <class T, class Allocator = std::allocator<T>>
using SVec48 = SmallVec<T, 47, Allocator>;

template <class T>
using RefVec24 = SVec24< std::reference_wrapper< T >>;

template <class T>
using RefVec32 = SVec32< std::reference_wrapper< T >>;

#endif