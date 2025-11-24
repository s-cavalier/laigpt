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
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG 0.25.0
)
FetchContent_MakeAvailable(xtensor)

file(GLOB SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(cppautograd ${SRC_FILES})

target_include_directories(cppautograd PRIVATE
    ${xtensor_SOURCE_DIR}/include
    ${xtl_SOURCE_DIR}/include
)

target_link_libraries(cppautograd PRIVATE xtensor xtl)

target_compile_options(cppautograd PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
