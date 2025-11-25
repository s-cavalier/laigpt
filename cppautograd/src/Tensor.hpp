#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <xtensor/xarray.hpp>
#include <vector>
#include <cstddef>
#include "PolyPtr.hpp"
#include <memory>

namespace ag {

    using f32Array = xt::xarray<float>;

    class GradientFunction {

        virtual f32Array forward(const std::vector< f32Array >& inputs) = 0;
        virtual std::vector<f32Array> backward( const f32Array& output, const std::vector<f32Array>& inputs ) = 0;

    };

    class DiagonalJacobianGradient : public GradientFunction {
        std::vector<f32Array> backward( const f32Array& output, const std::vector<f32Array>& inputs ) final override;
    };

    

    f32Array reduce_to_shape( const f32Array& grad, const std::vector<size_t>& shape );


    using GradFnPtr = PolyPtr<GradientFunction, 12>;

    class Tensor {
        std::unique_ptr<f32Array> values; // values must be on the heap because an f32Array is somehow 240 bytes each
        std::vector< Tensor > parents; // Tensors should be moved into parents
        GradFnPtr gradient;

    public:
        Tensor( bool track_gradient = true );
        explicit Tensor( f32Array&& array, bool track_gradient = true ); // Take ownership
        explicit Tensor( const f32Array& array, bool track_gradient = true ); // Copy

        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        inline f32Array& data();
        inline void addParent(Tensor&& t);
        inline void setGradientFunction( PolyPtr<GradientFunction, 12>&& grad_fn );

        void backpropagate();

        float gradient_value;
        bool track_gradient;
    };

}

#endif