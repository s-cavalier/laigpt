import numpy as np
from abc import ABCMeta, abstractmethod
from autograd.tensor import Tensor
import autograd.functions as F
from typing import Callable

class Module(metaclass=ABCMeta):
    def __init__(self):
        self.params: dict[str, Tensor] = {}
        self.submodules: dict[str, Module] = {}
        self.training: bool = True

    def parameters(self):
        for _, p in self.params.items(): yield p
        for _, m in self.submodules.items(): yield from m.parameters()

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



class WrapperModule(Module):
    def __init__(self, FunctionType: type[F.Function]):
        super().__init__()
        
        assert issubclass(FunctionType, F.Function)
        self.inst = FunctionType()
        
    def __call__(self, x):
        return self.inst(x)

class ReLU(WrapperModule):
    def __init__(self):
        super().__init__(F.ReLU)

class Sigmoid(WrapperModule):
    def __init__(self):
        super().__init__(F.Sigmoid)

class Softmax(WrapperModule):
    def __init__(self):
        super().__init__(F.Softmax)

class CrossEntropyLoss(Module):
    def __init__(self, targets: np.ndarray):
        super().__init__()
        self.inst = F.CrossEntropyLoss(targets)

    def __call__(self, logits: Tensor):
        return self.inst(logits)


# Some basics

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, init_fn: Callable[ [int, int], tuple[np.ndarray, np.ndarray] ] | None = None ):
        super().__init__()
        self.W = None 
        self.b = None
        if init_fn is not None: self.W, self.b = init_fn(in_dim, out_dim)
        else: 
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

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0,1)"
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0: return x

        keep_prob = 1.0 - self.p

        mask = (np.random.rand(*x.values.shape) < keep_prob).astype(x.values.dtype)
        mask /= keep_prob
        mask_tensor = Tensor(mask, track_grad=False)
        return x * mask_tensor

class Embedding(Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = Tensor(
            (np.random.randn(vocab_size, dim) / np.sqrt(vocab_size)).astype(np.float32),
            track_grad=True
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.weight[x]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.gamma = Tensor(np.ones((dim,), dtype=np.float32))
        self.beta  = Tensor(np.zeros((dim,), dtype=np.float32))

    def __call__(self, x: Tensor) -> Tensor:

        mean = F.Mean(axis=-1, keepdims=True)(x)

        var  = F.Variance(axis=-1, keepdims=True)(x)

        eps_tensor = Tensor(self.eps, track_grad=False)
        std = F.sqrt(var + eps_tensor)

        norm = (x - mean) / std

        reshape_shape = (1,) * (len(x.values.shape) - 1) + (self.dim,)
        gamma = self.gamma.reshape(reshape_shape)
        beta  = self.beta.reshape(reshape_shape)

        return norm * gamma + beta

class SelfAttentionHead(Module):
    ...