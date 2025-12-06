#include "Primitives.hpp"

#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>

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

    return ret;
}

void ag::Sine::func_impl( arena_f32arr& out_parameter, const parameter_container<arena_f32arr>& parameters ) const {
    xt::noalias(out_parameter) = xt::sin(parameters.front().get());
}
