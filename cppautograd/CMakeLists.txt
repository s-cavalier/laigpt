cmake_minimum_required(VERSION 3.15)
project(cppautograd)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG 0.7.5
)
FetchContent_MakeAvailable(xtl)


FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 11.0.0
)
FetchContent_MakeAvailable(xsimd)


FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.25.0
)
FetchContent_MakeAvailable(xtensor)

FetchContent_Declare(
    xtensor_blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
    GIT_TAG 0.20.0
)
FetchContent_MakeAvailable(xtensor_blas)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)

add_executable(cppautograd
    src/main.cpp
    src/TensorNetwork.cpp
)

target_include_directories(cppautograd PRIVATE
    ${xtl_SOURCE_DIR}/include
    ${xsimd_SOURCE_DIR}/include
    ${xtensor_SOURCE_DIR}/include
    ${xtensor_blas_SOURCE_DIR}/include
    ${OPENBLAS_INCLUDE_DIRS}
)

target_link_libraries(cppautograd PRIVATE
    xtensor
    xtl
    xsimd                             
    xtensor-blas
    ${OPENBLAS_LIBRARIES}
)

target_compile_options(cppautograd PRIVATE
    -Wall -Wextra -Wpedantic -O3 -march=native
)

target_compile_definitions(cppautograd PRIVATE XTENSOR_USE_XSIMD=1)
