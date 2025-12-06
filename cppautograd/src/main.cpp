#include <iostream>
#include <vector>

#include "TensorNetwork.hpp"
#include <xtensor/xio.hpp>
#include "SmallVector.hpp"

#include <algorithm>

#include <xtensor/xmath.hpp>

#include <xtensor/xnoalias.hpp>

template <class T>
using storage_container = SmallVec<T, 23, ag::TensorNetwork::BumpAllocator<T>>;

constexpr float PI = 3.141519f;

int main() {

    ag::TensorNetwork network;
    
    ag::arena_xarray<float> arr(network);
    arr = { 1. , 0.5 , 0.25 };

    auto new_arr = xt::sin(arr);
    xt::noalias(arr) = new_arr / xt::cos(arr);


    return 0;
}
