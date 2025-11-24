import numpy as np
from autograd.tensor import Tensor


def assert_same(a, b, msg=""):
    assert np.allclose(a, b), msg


def test_reverse_all_dims():
    print("Test: transpose() reverses all dims")
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), track_grad=False)

    y = x.transpose()
    expected = np.transpose(x.values)

    assert_same(y.values, expected)
    print("  OK")


def test_T_property():
    print("Test: .T property matches transpose()")
    x = Tensor(np.random.randn(5, 6).astype(np.float32), track_grad=False)

    y = x.T
    expected = np.transpose(x.values)

    assert_same(y.values, expected)
    print("  OK")


def test_two_axis_swap():
    print("Test: transpose(a, b) axis swap")
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), track_grad=False)

    y = x.transpose(1, 2)
    expected = np.transpose(x.values, (0, 2, 1))

    assert_same(y.values, expected)
    print("  OK")


def test_explicit_perm():
    print("Test: explicit permutation tuple")
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), track_grad=False)

    y = x.transpose((2, 0, 1))
    expected = np.transpose(x.values, (2, 0, 1))

    assert_same(y.values, expected)
    print("  OK")


def test_backward_reverse():
    print("Test: backward for reverse transpose")

    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x = Tensor(x_np, track_grad=True)

    y = x.transpose()      # reverse dims → shape (4,3,2)
    loss = y.sum()
    loss.backpropagate()

    # Gradient should be all ones, then inverse transpose() applied → same as reverse dims
    expected_grad = np.transpose(np.ones_like(y.values), (2,1,0))

    assert_same(x.grad_value, expected_grad)
    print("  OK")


def test_backward_two_axis_swap():
    print("Test: backward for axis swap")

    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x = Tensor(x_np, track_grad=True)

    y = x.transpose(1, 2)  # swap dims → shape (2,4,3)
    loss = y.sum()
    loss.backpropagate()

    # grad_y is all ones, inverse swap (1 <-> 2) gives:
    expected_grad = np.transpose(np.ones_like(y.values), (0, 2, 1))

    assert_same(x.grad_value, expected_grad)
    print("  OK")


def test_backward_full_perm():
    print("Test: backward for full permute")

    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x = Tensor(x_np, track_grad=True)

    perm = (2, 0, 1)
    y = x.transpose(perm)
    loss = y.sum()
    loss.backpropagate()

    inv = np.argsort(perm)
    expected_grad = np.transpose(np.ones_like(y.values), inv)

    assert_same(x.grad_value, expected_grad)
    print("  OK")


def test_grad_shape_consistency():
    print("Test: gradient shape matches input shape")

    x_np = np.random.randn(3, 4, 5, 6).astype(np.float32)
    x = Tensor(x_np, track_grad=True)

    y = x.transpose(0, 2)  # expand to full perm (0,2,1,3)
    loss = y.sum()
    loss.backpropagate()

    assert x.grad_value.shape == x_np.shape
    print("  OK")


def main():
    print("=== Testing Transpose ===\n")

    test_reverse_all_dims()
    test_T_property()
    test_two_axis_swap()
    test_explicit_perm()
    test_backward_reverse()
    test_backward_two_axis_swap()
    test_backward_full_perm()
    test_grad_shape_consistency()

    print("\n=== All Transpose tests passed ===")


if __name__ == "__main__":
    main()
