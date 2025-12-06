#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include "TensorNetwork.hpp"
#include "SmallVector.hpp"
#include <string>

namespace ag {

    using arena_f32arr = arena_xarray<float>;

    template <class T>
    using ref = std::reference_wrapper<T>;

    class Tensor {
        struct Node {
            arena_f32arr values;
            SVec24< ref<Node>, TensorNetwork::BumpAllocator< ref<Node> > > parents;

            Node( TensorNetwork& net ) : values(net), parents( net.get_allocator< ref<Node> >() ) {}
            Node( const arena_f32arr& arr, TensorNetwork& net ) : values(arr), parents( net.get_allocator< ref<Node> >()  ) {}
            Node( arena_f32arr&& arr, TensorNetwork& net ) : values( std::move(arr) ), parents( net.get_allocator< ref<Node> >()  ) {}
            std::string serialize_graph(bool include_values = false) const;
        
        };

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

    };





}

#endif