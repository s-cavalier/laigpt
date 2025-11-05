from core_math.tensor import Tensor, GradientFunction, DiagonalJacobianGradient
import numpy as np
from abc import ABCMeta, abstractmethod


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

class Sine(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            return np.cos(inputs[0])

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        return np.sin(x[0])
    
    gradient = Gradient()

    def get_gradient(self):
        return Sine.gradient

sin = Sine()

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

class Exp(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            return np.exp(inputs[0])
    
    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        return np.exp(x[0])
    
    gradient = Gradient()
    
    def get_gradient(self):
        return Exp.gradient
    
exp = Exp()

class Log(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            return 1.0 / x

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        return np.log(a)
    
    gradient = Gradient()

    def get_gradient(self):
        return Log.gradient

log = Log()
    
class Sqrt(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            return 0.5 / np.sqrt(x)

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        return np.sqrt(a)
    
    gradient = Gradient()

    def get_gradient(self):
        return Sqrt.gradient

sqrt = Sqrt()

class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    class Gradient(GradientFunction):
        def __init__(self, axis=None, keepdims=False):
            self.axis = axis
            self.keepdims = keepdims

        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            return np.ones_like(x)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            grad = output
            if not self.keepdims and self.axis is not None:
                grad = np.expand_dims(grad, axis=self.axis)
            return np.ones_like(x) * grad

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        return np.sum(a, axis=self.axis, keepdims=self.keepdims)

    def get_gradient(self):
        return Sum.Gradient(self.axis, self.keepdims)

class Mean(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    class Gradient(GradientFunction):
        def __init__(self, axis=None, keepdims=False):
            self.axis = axis
            self.keepdims = keepdims

        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            n = x.size if self.axis is None else x.shape[self.axis]
            return np.full_like(x, 1.0 / n)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            n = x.size if self.axis is None else x.shape[self.axis]
            grad = output / n
            if not self.keepdims and self.axis is not None:
                grad = np.expand_dims(grad, axis=self.axis)
            return np.ones_like(x) * grad

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        return np.mean(a, axis=self.axis, keepdims=self.keepdims)

    def get_gradient(self):
        return Mean.Gradient(self.axis, self.keepdims)