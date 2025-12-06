#ifndef __PRIMITIVES_HPP__
#define __PRIMITIVES_HPP__

#include "Tensor.hpp"
#include <concepts>


namespace ag {

    struct Function {
        template <class T>
        using parameter_container = SVec32< 
            ref<const T>, 
            TensorNetwork::BumpAllocator< ref<const T> >
        >;

    protected:
        virtual void func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& in_parameters ) const = 0;

    public:
        Tensor operator()( const parameter_container<Tensor>& parameters ) const;

        template <std::same_as<Tensor>... Ts> requires ( sizeof...(Ts) > 0 )
        Tensor operator()(const Ts&... args) {
            const Tensor& first = std::get<0>(std::tie(args...));
            parameter_container<Tensor> params( { args... }, first.net.get().get_allocator< ref< const Tensor> >()  );
            return (*this)(params);
        }

    };


    class Sine : public Function {
    protected:
        void func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const override;

    };

    inline Sine sin;

}

#endif