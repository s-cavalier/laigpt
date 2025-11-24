import numpy as np
from abc import ABCMeta, abstractmethod

class GradientFunction(metaclass=ABCMeta):
    
    @abstractmethod
    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Defines the basic function itself. For example, a Sine just implements cosine(inputs[0]).
        """
        
        pass

    @abstractmethod
    def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
        """
        Defines the consequence of dC/dx_{n-1} = dC/dx_n dx_n/dx_{n - 1}, i.e. the main back prop
        Should pretty much just define the interaction between the two terms on the right, since that might be a element-wise multiply or a matrix multiply
        """

        pass

class DiagonalJacobianGradient(GradientFunction):
    def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
        return output * self.forward(inputs[0])

def reduce_to_shape(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce broadcasted gradients to match the parameter shape."""

    while grad.ndim > len(target_shape):
        grad = np.sum(grad, axis=0)
    for i, (gdim, tdim) in enumerate(zip(grad.shape, target_shape)):
        if gdim != tdim and tdim == 1:
            grad = np.sum(grad, axis=i, keepdims=True)
    return grad.reshape(target_shape)

class Tensor:
    __slots__ = ("values", "track_grad", "grad_func", "grad_value", "parents")

    def __init__(self, values, track_grad=True):
        arr = np.asarray(values)
        if arr.dtype.kind == "f": arr = arr.astype(np.float32, copy=False) # Force f32
        self.values = arr
        self.track_grad = track_grad
        self.grad_func = None
        self.grad_value = None
        self.parents = []


    def serialize_graph(self):
        this = "Tensor{" f"{self.values}, {self.grad_func.__class__.__qualname__}" "}"
        if not self.parents:
            return this

        return f"[{ ', '.join(parent.serialize_graph() for parent in self.parents) }] -> {this}"

    def backpropagate(self, cost_value=None):
        if cost_value is None: cost_value = np.ones_like(self.values)

        if isinstance(cost_value, Tensor): raise RuntimeError( "Invalid gradient type passed to backpropagate: got Tensor; expected a NumPy array or something convertible to one." )

        if not isinstance(cost_value, np.ndarray):
            try: cost_value = np.asarray(cost_value)
            except Exception as e: raise RuntimeError( f"Invalid gradient type passed to backpropagate: {type(cost_value)} " f"(could not convert to ndarray: {e})" )

        if cost_value.dtype == object: raise RuntimeError( "Invalid gradient dtype=object passed to backpropagate " "(likely a Python object slipped into the graph)" )

        cost_value = reduce_to_shape(cost_value, self.values.shape)

        topo: list["Tensor"] = []
        visited: set[int] = set()

        def build(node: "Tensor"):
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            for parent in node.parents:
                build(parent)
            topo.append(node)

        build(self)

        for node in topo:
            node.grad_value = None

        self.grad_value = cost_value

        for node in reversed(topo):
            if node.grad_value is None or not node.track_grad:
                node.parents = []
                node.grad_func = None
                continue

            if node.grad_func is None or not node.parents:
                node.parents = []
                node.grad_func = None
                continue

            inputs = [parent.values for parent in node.parents]
            local_grads = node.grad_func.backward(node.grad_value, *inputs)

            if not isinstance(local_grads, tuple):
                local_grads = (local_grads,)

            for parent, gradient in zip(node.parents, local_grads):
                if gradient is None or not parent.track_grad: continue

                if not isinstance(gradient, np.ndarray):
                    raise RuntimeError( f"Gradient returned by {node.grad_func} for parent {parent} " f"is not a NumPy array (got {type(gradient)})" )

                if gradient.dtype == object: raise RuntimeError( f"Gradient returned by {node.grad_func} for parent {parent} " f"has dtype=object — illegal state" )

                gradient = reduce_to_shape(gradient, parent.values.shape)

                if parent.grad_value is None: parent.grad_value = gradient
                else: parent.grad_value += gradient

            node.parents = []
            node.grad_func = None




    def convert(value, track_grad = True):
        if isinstance(value, int): return Tensor([value], track_grad)
        if isinstance(value, float): return Tensor([value], track_grad)
        if isinstance(value, np.int64): return Tensor(value, track_grad)
        if isinstance(value, np.float64): return Tensor(value, track_grad)
        raise RuntimeError(f"Cannot convert value of type {type(value)} to a Tensor")

    def __add__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from autograd.primitives import add
        return add(self, other)

    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)


    def __sub__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from autograd.primitives import subtract
        return subtract(self, other)

    def __rsub__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)

        from autograd.primitives import subtract
        return subtract(other, self)


    def __mul__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from autograd.primitives import multiply
        return multiply(self, other)

    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)


    def __truediv__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from autograd.primitives import divide
        return divide(self, other)

    def __rtruediv__(self, other) -> 'Tensor':

        if not isinstance(other, Tensor): other = Tensor.convert(other)
        from autograd.primitives import divide
        return divide(other, self)


    def __matmul__(self, other) -> 'Tensor':
        # Avoid cyclic imports:
        from autograd.primitives import matrix_multiply
        return matrix_multiply(self, other)

    def __rmatmul__(self, other) -> 'Tensor':

        if not isinstance(other, Tensor): other = Tensor.convert(other)
        from autograd.primitives import matrix_multiply
        return matrix_multiply(other, self)


    def __neg__(self) -> 'Tensor':
        # Avoid cyclic imports:
        from autograd.primitives import neg
        return neg(self)


    def __pow__(self, other):
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from autograd.primitives import pow
        return pow(self, other)

    def __rpow__(self, other):
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        from autograd.primitives import pow
        return pow(other, self)
    
    def __getitem__(self, idx):
        from autograd.primitives import Slice, lookup
        
        if isinstance(idx, Tensor): return lookup(self, idx)
        
        return Slice(idx)(self)
    
    def reshape(self, shape: tuple[int, ...]):
        from autograd.primitives import Reshape

        return Reshape(shape)(self)
    
    def transpose(self, *axes):
        from autograd.primitives import Transpose

        if len(axes) == 0: axes = None

        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)): axes = tuple(axes[0])

        else: axes = tuple(axes)

        return Transpose(axes)(self)
    
    @property
    def T(self): return self.transpose()
    
    def sum(self, axis=None, keepdims=False):
        from autograd.functions import Sum
        
        return Sum(axis, keepdims)(self)


