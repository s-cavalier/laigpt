import numpy as np
from abc import ABCMeta, abstractmethod
from autograd.tensor import Tensor
import autograd.functions as F
from typing import Callable
from typing import Iterable

def serialize_parameters(module: "Module", prefix: str = "") -> dict:
    params = {}

    for name, tensor in module.params.items():
        key = prefix + name
        params[key] = tensor.values.copy()

    for name, submodule in module.submodules.items():
        sub_prefix = prefix + name + "."
        params.update(serialize_parameters(submodule, sub_prefix))

    return params

def deserialize_parameters(module: "Module", saved: dict, prefix: str = ""):
    for name, tensor in module.params.items():
        key = prefix + name

        if key not in saved: raise KeyError( f"Missing parameter '{key}' in saved file." )

        saved_arr = saved[key]
        if saved_arr.shape != tensor.values.shape:
            raise ValueError( f"Shape mismatch for '{key}': " f"expected {tensor.values.shape}, got {saved_arr.shape}" )
        tensor.values[...] = saved_arr

    for name, submodule in module.submodules.items():
        sub_prefix = prefix + name + "."
        deserialize_parameters(submodule, saved, sub_prefix)

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
                p.grad_value = None
        
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
        if not hasattr(self, "params"):
            super().__setattr__(name, value)
            return

        super().__setattr__(name, value)

        if isinstance(value, Module):
            self.submodules[name] = value

        elif isinstance(value, Tensor):
            if value.track_grad: self.params[name] = value

    def save(self, path: str):
        params = serialize_parameters(self)
        np.savez(path, **params)
        print(f"{self.__class__.__qualname__} saved {len(params)} tensors to {path}")

    def load(self, path: str):
        saved = np.load(path)

        saved_dict = {k: saved[k] for k in saved.files}

        deserialize_parameters(self, saved_dict)
        print(f"{self.__class__.__qualname__} Loaded {len(saved_dict)} tensors from {path}")

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

class GELU(WrapperModule):
    def __init__(self):
        super().__init__(F.GELU)

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

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules_list = []

        if modules is not None:
            for m in modules: 
                assert isinstance(m, Module)
                self.append(m)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __len__(self):
        return len(self._modules_list)

    def __iter__(self):
        return iter(self._modules_list)

    def append(self, module: Module):
        assert isinstance(module, Module), "ModuleList accepts only Module instances"

        idx = len(self._modules_list)
        self._modules_list.append(module)

        super().__setattr__(str(idx), module)

    def extend(self, modules):
        for m in modules: self.append(m)

    def __call__(self, *inputs):
        raise NotImplementedError("ModuleList should just be a helper to store modules ina list and avoid setattr issues; it shouldn't be called")

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, init_fn: Callable[[int, int], tuple[np.ndarray, np.ndarray]] | None = None, bias: bool = True):
        super().__init__()

        if init_fn is not None:
            W_np, b_np = init_fn(in_dim, out_dim)
            self.W = Tensor(W_np, track_grad=True)

            if bias: self.b = Tensor(b_np, track_grad=True)
            else: self.b = Tensor(np.zeros((1, out_dim), dtype=np.float32), track_grad=False)

        else:
            scale = 1.0 / np.sqrt(in_dim)
            self.W = Tensor( np.random.randn(in_dim, out_dim).astype(np.float32) * scale, track_grad=True )

            if bias: self.b = Tensor(np.zeros((1, out_dim), dtype=np.float32), track_grad=True)
            else: self.b = Tensor(np.zeros((1, out_dim), dtype=np.float32), track_grad=False)

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


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.gamma = Tensor(np.ones((dim,), dtype=np.float32))
        self.beta  = Tensor(np.zeros((dim,), dtype=np.float32))

        self._eps_tensor = Tensor(self.eps, track_grad=False)

    def __call__(self, x: Tensor) -> Tensor:
        mean = F.Mean(axis=-1, keepdims=True)(x)
        var  = F.Variance(axis=-1, keepdims=True)(x)
        std = F.sqrt(var + self._eps_tensor)

        norm = (x - mean) / std
        reshape_shape = (1,) * (len(x.values.shape) - 1) + (self.dim,)
        gamma = self.gamma.reshape(reshape_shape)
        beta  = self.beta.reshape(reshape_shape)

        return norm * gamma + beta