import numpy as np
from keras.datasets import mnist
from tqdm import tqdm
from autograd.tensor import Tensor
from autograd.functions import CrossEntropyLoss
from core.module import Sequential, Linear, ReLU
from core.optimize import Adam
from core.datastream import NumpyArraySource, MNISTData


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

    img_src = NumpyArraySource(train_x)
    label_src = NumpyArraySource(train_y)
    stream = MNISTData(img_src, label_src, batch_size=32, shuffle=True)

    # Define model and optimizer
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    )
    optimizer = Adam(model, lr=1e-3)

    epochs = 3

    for epoch in range(epochs):
        pbar = tqdm(stream, desc=f"Epoch {epoch+1}", colour="green")
        losses = []

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
                pbar.set_postfix({"loss": np.mean(losses[-100:])})

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

        acc = accuracy(model, test_x[:1000], test_y[:1000])
        print(f"Validation accuracy (1k test samples): {acc:.3f}")

    final_acc = accuracy(model, test_x, test_y)
    print(f"\nFinal Test Accuracy: {final_acc:.3f}")
