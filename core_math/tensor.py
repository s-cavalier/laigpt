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
        return (output * self.forward(inputs[0]),)


class Tensor:
    def __init__(self, values: np.ndarray, track_grad = True):
        self.values = np.asarray(values)
        self.track_grad = track_grad

        self.grad_func : GradientFunction | None = None
        self.grad_value : np.ndarray | None = None
        self.parents : list[Tensor] = []

    def serialize_graph(self):
        this = "Tensor{" f"{self.values}, {self.grad_func.__class__.__qualname__}" "}"
        if not self.parents:
            return this

        return f"[{ ', '.join(parent.serialize_graph() for parent in self.parents) }] -> {this}"

    def backpropagate(self, cost_value = None ):

        if cost_value is None: cost_value = np.ones_like(self.values)

        if self.grad_func is None:
            
            if self.grad_value is None: self.grad_value = cost_value
            else: self.grad_value += cost_value

            return

        inputs = [ parent.values for parent in self.parents ]

        local_gradients = self.grad_func.backward( cost_value, *inputs )
        if not isinstance(local_gradients, tuple):
            local_gradients = (local_gradients,) # type checking in case of returning a non-tuple

        for parent, gradient in zip(self.parents, local_gradients): parent.backpropagate(gradient)

    def convert(value, track_grad = True):
        if isinstance(value, int): return Tensor([value], track_grad)
        if isinstance(value, float): return Tensor([value], track_grad)
        raise RuntimeError(f"Cannot convert value of type {type(value)} to a Tensor")

    def __add__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from core_math.primitives import add
        return add(self, other)
    
    def __sub__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from core_math.primitives import subtract
        return subtract(self, other)
    
    def __mul__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from core_math.primitives import multiply
        return multiply(self, other)
    
    def __truediv__(self, other) -> 'Tensor':
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from core_math.primitives import divide
        return divide(self, other)
    
    def __matmul__(self, other) -> 'Tensor':

        # Avoid cyclic imports:
        from core_math.primitives import matrix_multiply
        return matrix_multiply(self, other)
    
    def __neg__(self) -> 'Tensor':

        # Avoid cyclic imports:
        from core_math.primitives import neg
        return neg(self)
    
    def __pow__(self, other):
        if not isinstance(other, Tensor): other = Tensor.convert(other)
        
        # Avoid cyclic imports:
        from core_math.primitives import pow
        return pow(self, other)