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

constexpr size_t operator""_kb( unsigned long long x ) {
    return x * 1024;
}

constexpr size_t operator""_mb( unsigned long long x ) {
    return x * 1024 * 1024;
}

int main() {
    ag::TensorNetwork net(1_mb);

    ag::Tensor x({ {1.0, 2.0}, {3.0, 4.0} }, net);

    ag::Tensor y = ag::cos(x) + x;

    std::cout << y.serialize_graph(true) << '\n';

    return 0;
}
