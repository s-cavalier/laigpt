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

class Embedding(Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # Each row corresponds to one token's embedding vector
        self.weight = Tensor(
            np.random.randn(vocab_size, dim) / np.sqrt(vocab_size)
        )

    def __call__(self, x: Tensor) -> Tensor:
        """
        x: Tensor of integer token IDs, shape (batch, seq_len)
        returns: Tensor of embeddings, shape (batch, seq_len, dim)
        """
        # Gather rows of the embedding matrix corresponding to token IDs
        embed_values = self.weight.values[x.values.astype(int)]
        return Tensor(embed_values)

class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = Tensor(np.ones((dim,)))
        self.beta = Tensor(np.zeros((dim,)))
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        """
        x: Tensor of shape (batch, seq_len, dim) or (seq_len, dim)
        returns: same shape
        """
        # Compute mean/variance over the last dimension (features)
        mean = np.mean(x.values, axis=-1, keepdims=True)
        var = np.var(x.values, axis=-1, keepdims=True)
        norm = (x.values - mean) / np.sqrt(var + self.eps)

        # Apply learnable affine parameters
        out = self.gamma.values * norm + self.beta.values
        return Tensor(out)
