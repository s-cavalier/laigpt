#include "Tensor.hpp"


ag::TensorNetwork::MemoryArena::MemoryArena(size_t bytes) : buffer{ (std::byte*)(::operator new(bytes)) }, size{bytes}, offset{0} {}
ag::TensorNetwork::MemoryArena::~MemoryArena() noexcept { ::operator delete(buffer); }
