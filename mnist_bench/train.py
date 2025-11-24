import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

from autograd.tensor import Tensor
from autograd.functions import CrossEntropyLoss
from core.module import Sequential, Linear, ReLU, LayerNorm, Dropout
from core.optimize import Adam
from core.datastream import NumpyArraySource, MNISTData


def preprocess():
    """Load and preprocess MNIST."""
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_x = train_x.reshape(-1, 784).astype(np.float32) / 255.0
    test_x  = test_x.reshape(-1, 784).astype(np.float32) / 255.0

    return (train_x, train_y), (test_x, test_y)


def he_init(in_dim, out_dim):
    W = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
    b = np.zeros((1, out_dim), dtype=np.float32)
    return W, b


def build_model():
    return Sequential(
        Linear(784, 256, init_fn=he_init),
        LayerNorm(256),
        ReLU(),
        Dropout(0.2),

        Linear(256, 256, init_fn=he_init),
        LayerNorm(256),
        ReLU(),
        Dropout(0.2),

        Linear(256, 128, init_fn=he_init),
        LayerNorm(128),
        ReLU(),
        Dropout(0.2),

        Linear(128, 10, init_fn=he_init)
    )


def accuracy(model, X, y, batch_size: int = 256) -> float:
    model.set_trainability(False)

    n = X.shape[0]
    correct = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = Tensor(X[start:end])
        logits = model(xb)
        preds = np.argmax(logits.values, axis=1)
        correct += np.sum(preds == y[start:end])

    return correct / n


if __name__ == "__main__":
    np.random.seed(42)

    (train_x, train_y), (test_x, test_y) = preprocess()

    batch_size = 128
    epochs = 10
    learning_rate = 1.5e-3

    train_stream = MNISTData(
        NumpyArraySource(train_x),
        NumpyArraySource(train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    model = build_model()
    optimizer = Adam(model, lr=learning_rate)

    for epoch in range(epochs):
        model.set_trainability(True)
        train_stream = iter(train_stream)

        losses = []
        pbar = tqdm(train_stream, desc=f"Epoch {epoch + 1}/{epochs}")

        for X_batch, Y_batch in pbar:
            X = Tensor(X_batch)
            Y = Y_batch

            optimizer.zero_gradients()

            logits = model(X)
            loss_fn = CrossEntropyLoss(Y)
            loss = loss_fn(logits)

            loss.backpropagate()
            optimizer.step()

            losses.append(loss.values)
            if len(losses) % 100 == 0:
                pbar.set_postfix({"loss": float(np.mean(losses[-100:]))})

        avg_loss = float(np.mean(losses))
        print(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

        val_acc = accuracy(model, test_x[:2000], test_y[:2000])
        print(f"Validation accuracy (2k test samples): {val_acc:.3f}")

    final_acc = accuracy(model, test_x, test_y)
    print(f"\nFinal Test Accuracy: {final_acc:.3f}")

    save_path = "mnist_model.npz"
    model.save(save_path)
    print(f"Saved trained MNIST model to {save_path}")
