import numpy as np
from autograd.tensor import Tensor
from core.module import Embedding 

def expected_grad(vocab, dim, ids_np):
    grad = np.zeros((vocab, dim), dtype=np.float32)
    flat = ids_np.reshape(-1)
    for idx in flat:
        grad[idx] += 1.0
    return grad


def test_forward():
    print("Test: forward")

    vocab = 5
    dim = 3
    np.random.seed(0)

    emb = Embedding(vocab, dim)
    ids_np = np.array([1, 3], dtype=np.int32)

    out = emb(Tensor(ids_np))

    assert out.values.shape == (2, dim)
    assert np.allclose(out.values, emb.weight.values[ids_np])
    print("  OK")


def test_backward_1d():
    print("Test: backward 1D indices")

    vocab, dim = 10, 4
    np.random.seed(1)

    emb = Embedding(vocab, dim)
    ids_np = np.array([3, 1, 3], dtype=np.int32)

    ids = Tensor(ids_np, track_grad=False)
    out = emb(ids)
    loss = out.sum()
    loss.backpropagate()

    expected = expected_grad(vocab, dim, ids_np)

    diff = np.abs(emb.weight.grad_value - expected).max()
    print("  max diff:", diff)
    assert diff == 0.0
    print("  OK")


def test_backward_2d():
    print("Test: backward 2D (batch x seq)")

    vocab, dim = 12, 5
    np.random.seed(2)

    emb = Embedding(vocab, dim)
    ids_np = np.array([[1, 2, 1], [0, 3, 3]], dtype=np.int32)

    ids = Tensor(ids_np, track_grad=False)
    out = emb(ids)
    loss = out.sum()
    loss.backpropagate()

    expected = expected_grad(vocab, dim, ids_np)

    diff = np.abs(emb.weight.grad_value - expected).max()
    print("  max diff:", diff)
    assert diff == 0.0
    print("  OK")


def test_only_used_rows_have_grad():
    print("Test: unused rows have zero grad")

    vocab, dim = 8, 3
    np.random.seed(3)

    emb = Embedding(vocab, dim)
    ids_np = np.array([2, 5, 2], dtype=np.int32)

    ids = Tensor(ids_np, track_grad=False)
    out = emb(ids)
    out.sum().backpropagate()

    expected = expected_grad(vocab, dim, ids_np)

    assert np.allclose(emb.weight.grad_value, expected)
    print("  OK")


def test_ids_get_no_grad():
    print("Test: indices receive no gradient")

    vocab, dim = 9, 3
    np.random.seed(4)

    emb = Embedding(vocab, dim)
    ids_np = np.array([1, 8, 2], dtype=np.int32)

    ids = Tensor(ids_np, track_grad=False)
    out = emb(ids)
    loss = out.sum()
    loss.backpropagate()

    assert ids.grad_value is None
    print("  OK")


def test_large_random():
    print("Test: large random stress test")

    vocab, dim = 50, 8
    B, T = 16, 10

    np.random.seed(5)
    emb = Embedding(vocab, dim)

    ids_np = np.random.randint(0, vocab, size=(B, T), dtype=np.int32)
    ids = Tensor(ids_np, track_grad=False)

    out = emb(ids)
    loss = out.sum()
    loss.backpropagate()

    expected = expected_grad(vocab, dim, ids_np)
    assert np.allclose(emb.weight.grad_value, expected)
    print("  OK")


def main():
    print("=== Testing Embedding Module ===\n")
    test_forward()
    test_backward_1d()
    test_backward_2d()
    test_only_used_rows_have_grad()
    test_ids_get_no_grad()
    test_large_random()
    print("\n=== All Embedding Module Tests Passed ===")


if __name__ == "__main__":
    main()
