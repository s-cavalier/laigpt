import numpy as np
from abc import ABCMeta, abstractmethod
from autograd.tensor import Tensor
import autograd.functions as F

class Module(metaclass=ABCMeta):
    def __init__(self):
        self.params: dict[str, Tensor] = {}
        self.submodules: dict[str, Module] = {}
        self.training: bool = True

    def parameters(self):
        for _, p in self.params.items():
            yield p
        for _, m in self.submodules.items():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            if p.grad_value is not None:
                p.grad_value[:] = 0
        
    def set_trainability(self, training_on = True):
        self.training = training_on
        for m in self.submodules.values():
            m.set_trainability(training_on)

    def add_module(self, name: str, module: "Module"):
        self.submodules[name] = module
        return module

    def add_param(self, name: str, tensor: Tensor):
        self.params[name] = tensor
        return tensor
    
    def __setattr__(self, name, value):
        # Dynamic submodule subparameter setup

        if not hasattr(self, "params"):
            super().__setattr__(name, value)
            return

        super().__setattr__(name, value)

        if isinstance(value, Tensor):
            self.params[name] = value
        elif isinstance(value, Module):
            self.submodules[name] = value



    @abstractmethod
    def __call__(self, *inputs: Tensor) -> Tensor:
        """Each module defines its forward call."""
        raise NotImplementedError

# Wrappers

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.inst = F.ReLU()

    def __call__(self, x):
        return self.inst(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.inst = F.Sigmoid()

    def __call__(self, x):
        return self.inst(x)

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = Tensor(np.random.randn(in_dim, out_dim))
        self.b = Tensor(np.zeros((1, out_dim)))

    def __call__(self, x: Tensor):
        return x @ self.W + self.b
    
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.submodules[f"layer_{i}"] = layer

    def __call__(self, x):
        for layer in self.submodules.values():
            x = layer(x)
        return x
