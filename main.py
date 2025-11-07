import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

from autograd.tensor import Tensor
from autograd.functions import CrossEntropyLoss
from core.module import Sequential, Linear, ReLU
from core.optimize import Adam


def preprocess():
    """Load and preprocess MNIST."""
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Normalize and flatten
    train_x = train_x.reshape(-1, 784).astype(np.float32) / 255.0
    test_x = test_x.reshape(-1, 784).astype(np.float32) / 255.0

    return (train_x, train_y), (test_x, test_y)


def accuracy(model, X, y):
    """Compute classification accuracy."""
    correct = 0
    for img, label in zip(X, y):
        out = model(Tensor(img))
        pred = np.argmax(out.values)
        correct += int(pred == label)
    return correct / len(y)


if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = preprocess()

    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    )

    optimizer = Adam(model, lr=1e-3)

    batch_size = 32
    epochs = 3

    num_batches = len(train_x) // batch_size

    for epoch in range(epochs):
        idx = np.random.permutation(len(train_x))
        train_x = train_x[idx]
        train_y = train_y[idx]

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", colour="green")
        losses = []

        for i in pbar:
            start, end = i * batch_size, (i + 1) * batch_size
            x_batch = Tensor(train_x[start:end])
            y_batch = train_y[start:end]

            # Reset gradients BEFORE forward
            optimizer.zero_gradients()

            # Forward pass
            logits = model(x_batch)

            # Loss
            loss_fn = CrossEntropyLoss(y_batch)
            loss = loss_fn(logits)

            # Backward + update
            loss.backpropagate()
            optimizer.step()

            losses.append(loss.values)
            if i % 100 == 0:
                pbar.set_postfix({"loss": np.mean(losses[-100:])})

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

        acc = accuracy(model, test_x[:1000], test_y[:1000])
        print(f"Validation accuracy (1k test samples): {acc:.3f}")

    final_acc = accuracy(model, test_x, test_y)
    print(f"\nFinal Test Accuracy: {final_acc:.3f}")
