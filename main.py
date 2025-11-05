import numpy as np
from autograd.tensor import Tensor, reduce_to_shape
import autograd.functions as F
from core.module import Linear, ReLU, Sigmoid, Sequential
from tqdm import tqdm

model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
    Sigmoid()
)

X = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y_true = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

loss_fn = F.MSE(Y_true)

lr = 0.1

pbar = tqdm(range(6_000_000), colour='red')

for epoch in pbar:
    Y_pred = model(X)
    loss = loss_fn(Y_pred)

    model.zero_grad()

    loss.backpropagate()

    for p in model.parameters():
        p.values -= lr * reduce_to_shape(p.grad_value, p.values.shape)

    if epoch % 10_000 == 0:
        pbar.set_postfix({"Loss":loss.values})

print("\nFinal predictions:")
print(Y_pred.values)
