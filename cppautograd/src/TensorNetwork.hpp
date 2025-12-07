#ifndef __TENSOR_NETWORK_HPP__
#define __TENSOR_NETWORK_HPP__

#include <cstddef>
#include <memory>
#include <functional>
#include <iostream>
#include <vector>
#include <type_traits>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xiterable.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>

// Really nice performant std::function implementation
#include "function.hpp"

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
                size_t bytes = n * sizeof(T);

                void* ptr = (void*)(buffer + offset);
                size_t space = size - offset;
                
                if ( !std::align( alignment, bytes, ptr, space ) ) {
                    throw std::runtime_error("TensorNetwork ran out of memory.");
                }
                
                offset = ( (std::byte*)(ptr) - buffer ) + bytes;
                return (T*)ptr;
            }

            void reset() noexcept { offset = 0; }
        };

        MemoryArena arena;

        friend class Tensor;

    public:
        template<class T>
        class BumpAllocator {
            MemoryArena* arena{nullptr};

            template<class> friend class BumpAllocator;

        public:
            using value_type = T;
            using pointer = T*;


            using propagate_on_container_copy_assignment = std::true_type;
            using propagate_on_container_move_assignment = std::true_type;
            using propagate_on_container_swap = std::true_type;
            using is_always_equal = std::true_type;

            BumpAllocator() noexcept = default;

            BumpAllocator(MemoryArena& a) noexcept : arena(&a) {}

            template<class U>
            BumpAllocator(const BumpAllocator<U>& other) noexcept : arena(other.arena) {}

            template<class U>
            struct rebind { using other = BumpAllocator<U>; };

            T* allocate(size_t n) {
                if (!arena) {
                    std::cerr << "WARNING: badly initialized bumpallocator.\n";
                    throw std::bad_alloc();
                }
                return arena->template allocate<T>(n);
            }

            void deallocate(T*, size_t) noexcept {
                // monotonic: no-op
            }

            bool operator==(const BumpAllocator& other) const noexcept {
                return arena == other.arena;
            }

            bool operator!=(const BumpAllocator& other) const noexcept {
                return arena != other.arena;
            }

            

        };



        TensorNetwork(size_t memory_allocated = 1024) : arena(memory_allocated) {}

        template <class T>
        BumpAllocator<T> get_allocator() {
            return BumpAllocator<T>(arena);
        }

    };

    template <class T>
    struct arena_raw_tensor_fields {
        using value_type = T;
        using index_type = std::size_t;

        using data_allocator = TensorNetwork::BumpAllocator<value_type>;
        using index_allocator = TensorNetwork::BumpAllocator<index_type>;

        using data_type = std::vector<value_type,   data_allocator>;
        using index_vec = std::vector<index_type,   index_allocator>;

        data_type data;
        index_vec shape;
        index_vec strides;
        index_vec backstrides;

        static constexpr xt::layout_type layout = xt::layout_type::row_major;

        explicit arena_raw_tensor_fields(TensorNetwork& net)
            : data(net.get_allocator<value_type>())
            , shape(net.get_allocator<index_type>())
            , strides(net.get_allocator<index_type>())
            , backstrides(net.get_allocator<index_type>())
        {}
    };

    template <class T>
    class arena_xarray;
}

namespace xt 
{
    template <class T>
    struct xcontainer_inner_types<ag::arena_xarray<T>> {
        using raw_fields_type = ag::arena_raw_tensor_fields<T>;

        using storage_type = typename raw_fields_type::data_type;

        using value_type = typename storage_type::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using inner_shape_type = typename raw_fields_type::index_vec;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type = inner_shape_type;

        using shape_type = inner_shape_type;
        using strides_type = inner_shape_type;
        using backstrides_type = inner_shape_type;

        static constexpr layout_type layout = raw_fields_type::layout;

        using temporary_type = xt::xarray<T>;
    };

    template <class T>
    struct xiterable_inner_types<ag::arena_xarray<T>> : xcontainer_iterable_types<ag::arena_xarray<T>> {};
}

namespace ag {

    template <class T>
    class arena_xarray : public xt::xcontainer<arena_xarray<T>> , public xt::xcontainer_semantic<arena_xarray<T>> {
        using self_type = arena_xarray<T>;
        using base_type = xt::xcontainer<self_type>;
        using semantic_base = xt::xcontainer_semantic<self_type>;
        using inner_types = xt::xcontainer_inner_types<self_type>;

        using raw_fields_type = arena_raw_tensor_fields<T>;

    public:
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;

        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        using inner_shape_type = typename inner_types::inner_shape_type;
        using inner_strides_type = typename inner_types::inner_strides_type;
        using inner_backstrides_type = typename inner_types::inner_backstrides_type;

        using temporary_type = typename inner_types::temporary_type;

        static constexpr xt::layout_type static_layout = inner_types::layout;

        friend class xt::xcontainer<arena_xarray<T>>;
        friend class xt::xiterable<arena_xarray<T>>;

        explicit arena_xarray(TensorNetwork& net) : m_raw(net) {}

        arena_xarray(TensorNetwork& net, const shape_type& shape, xt::layout_type l = static_layout) : m_raw(net) {
            resize(shape, l);
        }

        template <class E>
        arena_xarray(TensorNetwork& net, const xt::xexpression<E>& e) : xt::xcontainer<arena_xarray<T>>() , m_raw(net) {
            semantic_base::assign(e);
        }

        arena_xarray(TensorNetwork& net, std::initializer_list<T> init) : xt::xcontainer<arena_xarray<T>>() , m_raw(net) {
            assign_from_1d(init);
        }

        arena_xarray(TensorNetwork& net, std::initializer_list<std::initializer_list<T>> init) : xt::xcontainer<arena_xarray<T>>(), m_raw(net) {
            assign_from_2d(init);
        }

        using semantic_base::operator=;

        self_type& operator=(std::initializer_list<T> init) {
            assign_from_1d(init);
            return *this;
        }

        self_type& operator=(std::initializer_list<std::initializer_list<T>> init) {
            assign_from_2d(init);
            return *this;
        }

        xt::layout_type layout() const noexcept {
            return static_layout;
        }

        bool is_contiguous() const noexcept {
            return true;
        }

        void resize(const shape_type& shape) {
            if (m_raw.shape != shape) resize(shape, layout());
        }

        void resize(const shape_type& shape, xt::layout_type l) {
            m_raw.shape = shape;
            m_raw.strides.resize(shape.size());
            m_raw.backstrides.resize(shape.size());

            size_type data_size = xt::compute_strides(m_raw.shape, l, m_raw.strides, m_raw.backstrides);

            m_raw.data.resize(data_size);
        }

        void resize(const shape_type& shape, const strides_type& strides) {
            m_raw.shape = shape;
            m_raw.strides = strides;
            m_raw.backstrides.resize(shape.size());

            xt::adapt_strides(m_raw.shape, m_raw.strides, m_raw.backstrides);
            m_raw.data.resize(xt::compute_size(m_raw.shape));
        }

        self_type& assign_temporary(temporary_type&& tmp) {
            shape_type shape(m_raw.shape.get_allocator());
            auto const& tmp_shape = tmp.shape();

            std::size_t dim = tmp_shape.size();
            shape.resize(dim);
            for (std::size_t i = 0; i < dim; ++i) shape[i] = static_cast<size_type>(tmp_shape[i]);
            
            resize(shape, layout());

            auto dst = m_raw.data.begin();
            for (auto const& v : tmp) *dst++ = v;
            
            return *this;
        }

        template <class S, class = std::enable_if_t<!std::is_same<std::decay_t<S>, shape_type>::value>>
        void resize(const S& shape) {
            shape_type shp(m_raw.shape.get_allocator());
            shp.resize(shape.size());

            std::size_t i = 0;
            for (auto v : shape) shp[i++] = static_cast<size_type>(v);
            
            resize(shp);
        }

    protected:

        inner_shape_type& shape_impl() { return m_raw.shape; }
        const inner_shape_type& shape_impl() const { return m_raw.shape; }

        inner_strides_type& strides_impl() { return m_raw.strides; }
        const inner_strides_type& strides_impl() const { return m_raw.strides; }

        inner_backstrides_type& backstrides_impl() { return m_raw.backstrides; }
        const inner_backstrides_type& backstrides_impl() const { return m_raw.backstrides; }

        auto& storage_impl() { return m_raw.data; }
        const auto& storage_impl() const { return m_raw.data; }

        auto& data_impl() { return m_raw.data; }
        const auto& data_impl() const { return m_raw.data; }

    private:

        void assign_from_1d(std::initializer_list<T> init) {
            shape_type shape(m_raw.shape.get_allocator());
            shape.resize(1);
            shape[0] = static_cast<size_type>(init.size());

            resize(shape);
            std::copy(init.begin(), init.end(), m_raw.data.begin());
        }

        void assign_from_2d(std::initializer_list<std::initializer_list<T>> init) {
            size_type rows = static_cast<size_type>(init.size());
            size_type cols = rows ? static_cast<size_type>(init.begin()->size()) : size_type(0);

            shape_type shape(m_raw.shape.get_allocator());
            shape.resize(2);
            shape[0] = rows;
            shape[1] = cols;

            resize(shape);

            auto it = m_raw.data.begin();
            for (const auto& row : init) it = std::copy(row.begin(), row.end(), it);
            
        }

        raw_fields_type m_raw;
    };

    using xt::noalias;

    /*
    
    Some small wrappers around the function implementation.
    Can't find a good impl with an allocator type, but this is a fair workaround.
    
    */

    template<class>
    struct function_signature {};

    template<class Sig, class F>
    auto make_arena_function(ag::TensorNetwork& net, F&& f) {
        return make_arena_function(function_signature<Sig>{}, net, std::forward<F>(f));
    }

    template<class Ret, class... Args, class F>
    auto make_arena_function(function_signature<Ret(Args...)>, ag::TensorNetwork& net, F&& f) {

        using Functor = std::remove_cvref_t<F>;

        auto alloc = net.get_allocator<Functor>();
        Functor* p = alloc.allocate(1);
        std::construct_at(p, std::forward<F>(f));

        return func::function<Ret(Args...)>(
            [p](Args... a) -> Ret {
                return (*p)(std::forward<Args>(a)...);
            }
        );
    }

}


#endif
