#ifndef __PRIMITIVES_HPP__
#define __PRIMITIVES_HPP__

#include "Tensor.hpp"
#include <concepts>

#include <functional>
#include <vector>
#include <cstddef>


namespace ag {

    struct Function {

        Tensor operator()( const parameter_container<Tensor>& parameters ) const;

        // Compile-time convience op. Mostly for testing; a python binding has to take a runtime parameter set.
        template <std::same_as<Tensor>... Ts> requires ( sizeof...(Ts) > 0 )
        Tensor operator()(const Ts&... args) {
            const Tensor& first = std::get<0>(std::tie(args...));

            // propagate the allocator from the first allocator to the params vec
            parameter_container<Tensor> params( { args... }, first.net.get().get_allocator< ref< const Tensor> >()  );
            return (*this)(params);
        }

    private:
        Gradient::Function func;

    protected:
        /*
        
        Derived types of Function should be passing in a lambda to the Base constructor.
        If it is static or small enough to use SBO, then just the lambda should be passed in.
        Otherwise, a network must be passed in for the heap allocations.

        */

        template <class F> requires ( func::detail::is_inplace_allocated<std::remove_cvref_t<F>,std::allocator<std::remove_cvref_t<F>>>::value  )
        Function( F&& f ) : func(std::forward<F>(f)) {}


        template <class F>
        Function(TensorNetwork& net, F&& f) : func(make_arena_function<Gradient::FunctionArgs>(net, std::forward<F>(f))) {}

        
        virtual void func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& in_parameters ) const = 0;

    };

    #define AG_STATIC_FUNC_PROTOTYPE( class_name, inst_name, gradient_lambda_internal )        \
    class class_name final : public Function {  \
        class_name() : Function(      \
            [] (TensorNetwork& net,  \
                const arena_f32arr& output,      \
                [[maybe_unused]] const parameter_container<arena_f32arr>& inputs)  \
                -> Gradient::ReturnType gradient_lambda_internal  \
        ) {}  \
    protected: \
        void func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const override; \
    public:                                                                                \
        static class_name& instance() {                                                    \
            static class_name inst;                                                        \
            return inst;                                                                   \
        }                                                                                  \
    };                                                                                     \
    inline class_name& inst_name = class_name::instance()


    AG_STATIC_FUNC_PROTOTYPE( Sine, sin, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        xt::noalias( ret.front() ) = output * xt::cos( inputs.front().get() );
        return ret;
    } );

    AG_STATIC_FUNC_PROTOTYPE( Cosine, cos, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        xt::noalias( ret.front() ) = output * -xt::sin( inputs.front().get() );
        return ret;
    } );
    
    AG_STATIC_FUNC_PROTOTYPE( Tangent, tan, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        xt::noalias( ret.front() ) = output / xt::pow<2>(xt::cos( inputs.front().get() ));
        return ret;
    } );

    AG_STATIC_FUNC_PROTOTYPE( Negate, neg, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        xt::noalias( ret.front() ) = -output;
        return ret;
    } );

    AG_STATIC_FUNC_PROTOTYPE( Add, add, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        ret.emplace_back(net);
        xt::noalias( ret[0] ) = output;
        xt::noalias( ret[1] ) = output;
        return ret;
    } );

    AG_STATIC_FUNC_PROTOTYPE( Subtract, sub, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        ret.emplace_back(net);
        xt::noalias( ret[0] ) = output;
        xt::noalias( ret[1] ) = -output;
        return ret;
    } );

    AG_STATIC_FUNC_PROTOTYPE( Multiply, mul, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        ret.emplace_back(net);
        xt::noalias( ret[0] ) = output * inputs[1].get();
        xt::noalias( ret[1] ) = output * inputs[0].get();
        return ret;
    } );

    AG_STATIC_FUNC_PROTOTYPE( Divide, div, {
        Gradient::ReturnType ret{ net.get_allocator< arena_f32arr >() };
        ret.emplace_back(net);
        ret.emplace_back(net);
        const auto& a = inputs[0].get();
        const auto& b = inputs[1].get();
        xt::noalias( ret[0] ) = output / b;
        xt::noalias( ret[1] ) = output * ( -a / ( b * b ) ) ;
        return ret;
    } );

}

#endif