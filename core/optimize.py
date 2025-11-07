import numpy as np
from autograd.tensor import Tensor, reduce_to_shape
from .module import Module
from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):
    def __init__(self, module : Module, lr : float = 0.3 ):
        self.module = module
        self.lr = lr

    def zero_gradients(self):
        self.module.zero_grad()

    @abstractmethod
    def step(self):
        pass


class SimpleGradientDescent(Optimizer):
    def __init__(self, module: Module, lr : float = 0.3):
        super().__init__(module, lr)
    
    def step(self):
        for p in self.module.parameters():
            p.values -= self.lr * reduce_to_shape(p.grad_value, p.values.shape)

class Adam(Optimizer):
    def __init__(
        self,
        module: Module,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps=1e-8
    ):
        super().__init__(module, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0

        # Initialize moment buffers for each parameter
        self.params = list(module.parameters())
        self.m = [np.zeros_like(p.values) for p in self.params]
        self.v = [np.zeros_like(p.values) for p in self.params]

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            if p.grad_value is None:
                continue

            g = reduce_to_shape(p.grad_value, p.values.shape)
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g * g)

            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)

            p.values -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


