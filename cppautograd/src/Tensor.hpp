#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include "TensorNetwork.hpp"
#include "SmallVector.hpp"
#include <string>

namespace ag {

    using arena_f32arr = arena_xarray<float>;

    template <class T>
    using ref = std::reference_wrapper<T>;

    /*
    GradientFunction typedef.
    Since gradientfunction just needs to define the result of
    $dC / dx_{n-1} = dC / dx_n dx_{n - 1}$,
    we only need a single callable and the ability to store some state 
    in some cases. The first parameter is a TensorNetwork. Use this for the
    returning vector to allocate exactly the amount of memory that you need
    for the return vector and the corresponding arena_f32arr values.
    */

    template <class T>
    using parameter_container = SVec32< 
        ref<const T>, 
        TensorNetwork::BumpAllocator< ref<const T> >
    >;

    /*
    
    Gradient typedefs. Uses a special std::function impl for speedups.
    
    */
    struct Gradient {

        using ReturnType = std::vector<arena_f32arr, TensorNetwork::BumpAllocator< arena_f32arr >>;
        using FunctionArgs = ReturnType (TensorNetwork&, const arena_f32arr&, const parameter_container<arena_f32arr>&);
        using Function = func::function<FunctionArgs>;

        Gradient() = delete;
    };

    class Tensor {
        struct Node {
            arena_f32arr values;
            SVec24< ref<Node>, TensorNetwork::BumpAllocator< ref<Node> > > parents;
            Gradient::Function grad_func;
            arena_f32arr grad_values;
            bool track_gradient;

            Node( TensorNetwork& net ) : values(net), parents( net.get_allocator< ref<Node> >() ) {}
            Node( const arena_f32arr& arr, TensorNetwork& net ) : values(arr), parents( net.get_allocator< ref<Node> >()  ) {}
            Node( arena_f32arr&& arr, TensorNetwork& net ) : values( std::move(arr) ), parents( net.get_allocator< ref<Node> >()  ) {}
            std::string serialize_graph(bool include_values = false) const;
        
        };

        static constexpr size_t test = sizeof(Node);

        ref<TensorNetwork> net;
        ref<Node> node;

        // Private in favor of get_ref() and get_clone()
        Tensor(const Tensor& other) = default;

        // Private in favor of get_ref() and get_clone()
        Tensor& operator=(const Tensor& other) = default;

        Node& operator*();
        const Node& operator*() const;
        Node* operator->();
        const Node* operator->() const;

        friend class Function;

    public:

        Tensor( std::initializer_list<float> init_values, TensorNetwork& network );
        Tensor( std::initializer_list< std::initializer_list<float> > init_values, TensorNetwork& network );
        Tensor( const arena_f32arr& array, TensorNetwork& network );
        Tensor( arena_f32arr&& array, TensorNetwork& network );

        Tensor( Tensor&& other ) = default;
        Tensor& operator=(Tensor&& other) = default;

        arena_f32arr& values();
        const arena_f32arr& values() const;

        Tensor get_ref();

        Tensor get_clone() const;

        std::string serialize_graph(bool include_values = false) const;

        void backpropagate(const arena_f32arr* cost);

        Tensor operator-() const;

    };

    Tensor operator+(const Tensor& a, const Tensor& b);
    Tensor operator-(const Tensor& a, const Tensor& b);
    Tensor operator*(const Tensor& a, const Tensor& b);
    Tensor operator/(const Tensor& a, const Tensor& b);

}

#endif