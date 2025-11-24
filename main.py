import numpy as np
from autograd.tensor import Tensor
from core.gpt import TransformerBlock


def test_transformer_block_forward():
    print("Test: TransformerBlock forward")

    dim = 32
    heads = 4
    b, s = 2, 10

    x = Tensor(np.random.randn(b, s, dim).astype(np.float32), track_grad=True)
    block = TransformerBlock(dim, heads, ffn_expansion=4, attn_bias=False, ffn_bias=False)

    out = block(x)
    assert out.values.shape == (b, s, dim)
    print("  OK")


def test_transformer_block_backward():
    print("Test: TransformerBlock backward")

    dim = 32
    heads = 4
    b, s = 2, 10

    x = Tensor(np.random.randn(b, s, dim).astype(np.float32), track_grad=True)
    block = TransformerBlock(dim, heads, ffn_expansion=4, attn_bias=False, ffn_bias=False)

    out = block(x)
    loss = out.sum()
    loss.backpropagate()

    # Input should have gradient
    assert x.grad_value is not None, "x has no gradient"

    # Check some key params got gradients
    assert block.mha.qkv_proj.W.grad_value is not None, "MHA qkv_proj.W has no grad"
    assert block.mha.out_proj.W.grad_value is not None, "MHA out_proj.W has no grad"
    assert block.ffn.fc1.W.grad_value is not None, "FFN fc1.W has no grad"
    assert block.ffn.fc2.W.grad_value is not None, "FFN fc2.W has no grad"

    print("  OK")


def test_transformer_block_with_mask():
    print("Test: TransformerBlock with causal mask")

    dim = 16
    heads = 2
    b, s = 2, 6

    x = Tensor(np.random.randn(b, s, dim).astype(np.float32), track_grad=True)
    block = TransformerBlock(dim, heads)

    # Build a simple causal mask: (s, s) with -inf above diagonal
    mask_np = np.triu(np.ones((s, s), dtype=np.float32), k=1) * -1e9
    mask = Tensor(mask_np, track_grad=False)

    out = block(x, mask=mask)
    assert out.values.shape == (b, s, dim)

    loss = out.sum()
    loss.backpropagate()
    assert x.grad_value is not None

    print("  OK")


def main():
    print("=== Testing TransformerBlock ===")
    test_transformer_block_forward()
    test_transformer_block_backward()
    test_transformer_block_with_mask()
    print("\n=== All TransformerBlock tests passed ===")


if __name__ == "__main__":
    main()
