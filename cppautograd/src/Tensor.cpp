#include "Tensor.hpp"


ag::f32Array ag::reduce_to_shape( const ag::f32Array& grad, const std::vector<size_t>& target_shape) {
    ag::f32Array g = grad;

    while (g.dimension() > target_shape.size()) g = xt::sum(g, {0});
    
    for (size_t i = 0; i < target_shape.size(); ++i) {
        auto gdim = g.shape(i);
        auto tdim = target_shape[i];

        if (gdim != tdim && tdim == 1) {
            g = xt::sum(g, {i}, xt::keep_dims);
        }
    }

    return xt::reshape_view(g, target_shape);
}

ag::Tensor::Tensor( bool track_gradient ) : 
    values( std::make_unique<f32Array>() ),
    parents(),
    gradient( ag::GradFnPtr::emplace<GradientFunction>() ),
    gradient_value{0},
    track_gradient{track_gradient}
    {}

