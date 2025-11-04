import numpy as np
from core_math.tensor import Tensor
from core_math.primitives import sin, add

x = Tensor(np.array([2.0, -3.0]))
y = -x
y.backpropagate()
print("x.values =", x.values)
print("y.values =", y.values)
print("x.grad_value =", x.grad_value)