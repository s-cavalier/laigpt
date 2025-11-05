from autograd.tensor import Tensor, GradientFunction, DiagonalJacobianGradient
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Callable

class Function(metaclass=ABCMeta):
    @abstractmethod
    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        """
        This should just be the basic implementation of the function. The super handles the gradient attaching logic.
        I.e., sin should just call sin.
        """
        pass

    @abstractmethod
    def get_gradient(self) -> GradientFunction:
        """
        This should return the function used for gradient calculation.
        The function returned should have the first argument be the input value, and the second be the output value.
        So Sine would just do a component-wise cosine on the first argument and then a component-wise multiply by the second arg.
        """
        pass

    def __call__(self, *xs: Tensor):

        inputs = [x.values for x in xs]
        result = self.func_impl(*inputs)

        ret = Tensor(result, xs[0].track_grad)

        if ret.track_grad:
            ret.grad_func = self.get_gradient()
            ret.parents.extend(xs)

        return ret

class Neg(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            return -np.ones_like(x)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            return output * -1

        def __repr__(self):
            return "Neg.Gradient"

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        return -a

    gradient = Gradient()

    def get_gradient(self):
        return Neg.gradient

neg = Neg()

class Add(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            a, b = inputs[0], inputs[1]
            return np.ones_like(a), np.ones_like(b)
        
        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
            return output, output
        
    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        a, b = x
        return a + b
    
    gradient = Gradient()

    def get_gradient(self):
        return Add.gradient
    
add = Add()

class Sub(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            a, b = inputs
            return np.ones_like(a), -np.ones_like(b)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
            return output, -output

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        a, b = x
        return a - b

    gradient = Gradient()

    def get_gradient(self):
        return Sub.gradient

subtract = Sub()

class Mul(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            a, b = inputs
            return b, a

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
            a, b = inputs
            return output * b, output * a

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        a, b = x
        return a * b

    gradient = Gradient()

    def get_gradient(self):
        return Mul.gradient

multiply = Mul()

class Div(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            a, b = inputs
            return 1 / b, -a / (b ** 2)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
            a, b = inputs
            return output * (1 / b), output * (-a / (b ** 2))

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        a, b = x
        return a / b

    gradient = Gradient()

    def get_gradient(self):
        return Div.gradient

divide = Div()

class MatMul(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            a, b = inputs
            return b.T, a.T
        
        def backward(self, grad_output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray]:
            a, b = inputs
            dA = grad_output @ b.T
            dB = a.T @ grad_output
            return dA, dB

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        a, b = x
        return a @ b

    gradient = Gradient()
    def get_gradient(self):
        return MatMul.gradient

matrix_multiply = MatMul()

class Pow(Function):
    class Gradient(GradientFunction):
        def forward(self, *inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a, b = inputs
            grad_a = b * np.power(a, b - 1)
            grad_b = np.power(a, b) * np.log(a)
            return grad_a, grad_b

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a, b = inputs
            grad_a = output * b * np.power(a, b - 1)
            grad_b = output * np.power(a, b) * np.log(a)
            return grad_a, grad_b

    def func_impl(self, *inputs: np.ndarray) -> np.ndarray:
        a, b = inputs
        return np.power(a, b)
    
    gradient = Gradient()

    def get_gradient(self):
        return Pow.gradient

pow = Pow()


# Views

class ViewOperation(Function):
    """
    View operation classifies things like reshape, transpose, etc.
    Since there's not really a reasonable gradient for such a thing (generally), 
    you just need to specify a "reverse" operation in the back propagation in a subclass via super().__init__( reverse operation callable ).
    For a transpose, this would be another transpose, for example.
    """

    class Gradient(GradientFunction):
        def __init__(self, reverse_operation: Callable[ [np.ndarray], np.ndarray] ):
            self.reverse_operation = reverse_operation

        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            raise RuntimeError("Forward should not be called on a ViewGradient operation")
        
        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            return self.reverse_operation(output)
        
    def __init__(self, reverse_operation: Callable[ [np.ndarray], np.ndarray ] ):
        self.gradient = ViewOperation.Gradient(reverse_operation)

    def get_gradient(self):
        return self.gradient
    
class Reshape(ViewOperation):
    def __init__(self, new_shape: tuple[int, ...]):
        self.original_shape = None
        self.new_shape = new_shape

        def reverse( x: np.ndarray ) -> np.ndarray:
            assert self.original_shape is not None, "Tried to call reverse() before ever running it initially"
            return np.reshape(x, self.original_shape)
        
        super().__init__(reverse)

    def func_impl(self, *x: np.ndarray):
        self.original_shape = x[0].shape
        return np.reshape( x[0], self.new_shape )
    
class Transpose(ViewOperation):
    def __init__(self, axes=None):
        self.axes = axes
        self.inverse_axes = None

        def reverse( x: np.ndarray ) -> np.ndarray: return np.transpose(x, self.inverse_axes)

        super().__init__(reverse)

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        self.inverse_axes = ( np.argsort(self.axes) if self.axes is not None else None )
        return np.transpose(x[0], self.axes)
    

    