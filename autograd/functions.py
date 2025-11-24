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
    
class Variance(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    class Gradient(GradientFunction):
        def __init__(self, axis=None, keepdims=False):
            self.axis = axis
            self.keepdims = keepdims

        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            x = inputs[0]
            m = np.mean(x, axis=self.axis, keepdims=True)
            return 2 * (x - m) / (x.shape[self.axis] if self.axis is not None else x.size)

        def backward(self, output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            x = inputs[0]

            m = np.mean(x, axis=self.axis, keepdims=True)
            n = (x.shape[self.axis] if self.axis is not None else x.size)

            grad = 2 * (x - m) / n

            if not self.keepdims and self.axis is not None:
                grad_output = np.expand_dims(output, axis=self.axis)
            else:
                grad_output = output

            return grad * grad_output

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        m = np.mean(a, axis=self.axis, keepdims=True)
        sq = (a - m) ** 2
        return np.mean(sq, axis=self.axis, keepdims=self.keepdims)

    def get_gradient(self):
        return Variance.Gradient(self.axis, self.keepdims)
    
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

class Softmax(Function):
    def __init__(self, axis=-1):
        self.axis = axis

    class Gradient(GradientFunction):
        def __init__(self, axis):
            self.axis = axis

        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            e = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
            s = e / np.sum(e, axis=self.axis, keepdims=True)
            return s

        def backward(self, grad_output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            (x,) = inputs
            s = self.forward(x)
            dot = np.sum(grad_output * s, axis=self.axis, keepdims=True)
            return s * (grad_output - dot)

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (a,) = x
        e = np.exp(a - np.max(a, axis=self.axis, keepdims=True))
        return e / np.sum(e, axis=self.axis, keepdims=True)

    def get_gradient(self):
        return Softmax.Gradient(self.axis)

softmax = Softmax()

class CrossEntropyLoss(Function):

    class Gradient(GradientFunction):
        def __init__(self, targets: np.ndarray, axis=-1):
            self.targets = np.asarray(targets)
            self.axis = axis

        def forward(self, *inputs: np.ndarray) -> np.ndarray:
            (logits,) = inputs
            shifted = logits - np.max(logits, axis=self.axis, keepdims=True)
            e = np.exp(shifted)
            probs = e / np.sum(e, axis=self.axis, keepdims=True)

            if self.targets.ndim == 1:
                n = logits.shape[0]
                return -np.mean(np.log(probs[np.arange(n), self.targets.astype(int)] + 1e-12))
            else:
                return -np.mean(np.sum(self.targets * np.log(probs + 1e-12), axis=self.axis))

        def backward(self, grad_output: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
            (logits,) = inputs
            shifted = logits - np.max(logits, axis=self.axis, keepdims=True)
            e = np.exp(shifted)
            probs = e / np.sum(e, axis=self.axis, keepdims=True)

            n = logits.shape[0]
            grad_logits = probs.copy()
            if self.targets.ndim == 1:
                grad_logits[np.arange(n), self.targets.astype(int)] -= 1
            else:
                grad_logits -= self.targets

            grad_logits /= n
            grad_logits *= grad_output
            return grad_logits

    def __init__(self, targets: np.ndarray, axis=-1):
        assert isinstance(targets, np.ndarray)
        self.targets = np.asarray(targets)
        self.axis = axis
        self.gradient = CrossEntropyLoss.Gradient(self.targets, axis)

    def func_impl(self, *x: np.ndarray) -> np.ndarray:
        (logits,) = x
        shifted = logits - np.max(logits, axis=self.axis, keepdims=True)
        e = np.exp(shifted)
        probs = e / np.sum(e, axis=self.axis, keepdims=True)

        if self.targets.ndim == 1:
            n = logits.shape[0]
            return -np.mean(np.log(probs[np.arange(n), self.targets.astype(int)] + 1e-12))
        else:
            return -np.mean(np.sum(self.targets * np.log(probs + 1e-12), axis=self.axis))

    def get_gradient(self):
        return self.gradient
    
    
