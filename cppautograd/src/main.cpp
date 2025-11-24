#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>

int main()
{
    xt::xarray<double> a = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    auto b = xt::sin(a);

    // Broadcasting example
    xt::xarray<double> c = {10.0, 20.0, 30.0};
    auto d = a + c;

    std::cout << "a =\n" << a << "\n\n";
    std::cout << "sin(a) =\n" << b << "\n\n";
    std::cout << "a + c =\n" << d << "\n\n";

    return 0;
}
