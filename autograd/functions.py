import numpy as np
from autograd.primitives import Function, GradientFunction, DiagonalJacobianGradient


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
    
class MSE(Function):
    class Gradient(GradientFunction):
        def __init__(self, target: np.ndarray):
            self.target = np.asarray(target)

        def forward(self, x: np.ndarray):
            diff = x - self.target
            return np.mean(diff ** 2)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            n = x.size
            diff = x - self.target
            return output * (2.0 / n) * diff

    def __init__(self, target: np.ndarray):
        assert isinstance(target, np.ndarray)

        self.target = np.asarray(target)
        self.gradient = self.Gradient(self.target)

    def func_impl(self, x: np.ndarray):
        diff = x - self.target
        return np.mean(diff ** 2)

    def get_gradient(self):
        return self.gradient

class Sigmoid(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, input: np.ndarray):
            s = 1 / (1 + np.exp(-input))
            return s * (1 - s)

    def func_impl(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    gradient = Gradient()

    def get_gradient(self):
        return Sigmoid.gradient
    
sigmoid = Sigmoid()

class Tanh(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, input: np.ndarray):
            return 1 - np.tanh(input) ** 2

    def func_impl(self, x: np.ndarray):
        return np.tanh(x)

    gradient = Gradient()

    def get_gradient(self):
        return Tanh.gradient
    
tanh = Tanh()

class ReLU(Function):
    class Gradient(DiagonalJacobianGradient):
        def forward(self, input: np.ndarray):
            return (input > 0).astype(float)

    def func_impl(self, x: np.ndarray):
        return np.maximum(0, x)

    gradient = Gradient()

    def get_gradient(self):
        return ReLU.gradient

relu = ReLU()