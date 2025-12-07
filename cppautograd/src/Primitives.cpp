#include "Primitives.hpp"

#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>

#include <iostream>
#include <xtensor/xio.hpp>

ag::Tensor ag::Function::operator()(const parameter_container<Tensor>& parameters) const {
    
    // grab an allocator from one of the tensors
    auto& network = parameters.front().get().net.get();

    parameter_container<arena_f32arr> inputs( network.get_allocator< ref<const arena_f32arr> >() );
    inputs.reserve( parameters.size() );

    for ( const auto& tensor : parameters ) inputs.push_back( tensor.get()->values ); 

    arena_f32arr array(network);

    func_impl( array, inputs );

    Tensor ret( std::move(array), network );

    ret->parents.reserve( inputs.size() );
    for ( const auto& tensor : parameters ) ret->parents.push_back( tensor.get().node );

    // Need to investigate if this will ever cause a heap-alloc
    ret->grad_func = func;


    return ret;
}

void ag::Sine::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = xt::sin(parameters.front().get());
}

void ag::Cosine::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = xt::cos(parameters.front().get());
}

void ag::Tangent::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = xt::tan(parameters.front().get());
}

void ag::Negate::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = -parameters.front().get();
}

void ag::Add::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = parameters[0].get() + parameters[1].get();
}

void ag::Subtract::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = parameters[0].get() - parameters[1].get();
}

void ag::Multiply::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = parameters[0].get() * parameters[1].get();
}

void ag::Divide::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = parameters[0].get() / parameters[1].get();
}