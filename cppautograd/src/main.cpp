#include <iostream>
#include <vector>

#include "TensorNetwork.hpp"
#include <xtensor/xio.hpp>
#include "SmallVector.hpp"

#include <algorithm>

#include "Primitives.hpp"

template <class T>
using storage_container = SmallVec<T, 23, ag::TensorNetwork::BumpAllocator<T>>;

constexpr float PI = 3.141519f;

int main() {

    ag::TensorNetwork network;
    
    ag::Tensor x( { {1. , 0.5} , {0.25, 0.125} }, network );

    ag::Tensor y = ag::sin( x );

    std::cout << y.serialize_graph(false) << '\n';

    return 0;
}
