import numpy as np
from autograd.tensor import Tensor
from autograd.functions import attention, gelu
from core.module import Module, Linear, LayerNorm, ModuleList
from core.tokenizer import Tokenizer

class Embedding(Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = Tensor(
            (np.random.randn(vocab_size, dim) / np.sqrt(vocab_size)).astype(np.float32),
            track_grad=True
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.weight[x]
    
class PositionalEmbedding(Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.weight = Tensor(
            np.random.randn(max_seq_len, dim).astype(np.float32) / np.sqrt(dim),
            track_grad=True
        )

    def __call__(self, seq_len: int) -> Tensor:
        assert seq_len <= self.max_seq_len, ( f"Requested seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        pos_ids_np = np.arange(seq_len, dtype=np.int32)[None, :]
        pos_ids = Tensor(pos_ids_np, track_grad=False)

        out = self.weight[pos_ids]
        return out

class MultiHeadAttention(Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = Linear(dim, 3 * dim, bias=bias)
        self.out_proj = Linear(dim, dim, bias=bias)

    def _split_heads(self, x: Tensor) -> Tensor:

        b, s, d = x.values.shape
        h = self.num_heads
        hd = self.head_dim

        x = x.reshape((b, s, h, hd))
        x = x.transpose(0, 2, 1, 3)
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:

        b, h, s, hd = x.values.shape
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape((b, s, h * hd))
        return x

    def __call__(self, x: Tensor, mask: Tensor | None = None) -> Tensor:

        qkv = self.qkv_proj(x)
        D = self.dim
        q = qkv[..., :D]
        k = qkv[..., D:2*D]
        v = qkv[..., 2*D:]

        q = self._split_heads(q)    
        k = self._split_heads(k)
        v = self._split_heads(v)

        if mask is not None:
            m = mask.values.ndim
            if m == 2:
                s1, s2 = mask.values.shape
                mask = mask.reshape((1, 1, s1, s2))
            elif m == 3:
                b, s1, s2 = mask.values.shape
                mask = mask.reshape((b, 1, s1, s2))

        attn_out = attention(q, k, v, mask)  
        out = self._merge_heads(attn_out)     
        out = self.out_proj(out)              

        return out

class FeedForward(Module):
    def __init__(self, dim: int, expansion: int = 4, bias: bool = False):
        super().__init__()

        hidden_dim = dim * expansion

        self.fc1 = Linear(dim, hidden_dim, bias=bias)
        self.fc2 = Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(gelu(self.fc1(x)))

class TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        attn_bias: bool = False,
        ffn_bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.mha = MultiHeadAttention(dim, num_heads, bias=attn_bias)
        self.ffn = FeedForward(dim, expansion=ffn_expansion, bias=ffn_bias)

    def __call__(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attn_out = self.mha(self.ln1(x), mask)  
        x = x + attn_out

        ffn_out = self.ffn(self.ln2(x))         
        x = x + ffn_out

        return x


class GPT(Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len,
                 ffn_expansion=4, attn_bias=False, ffn_bias=False):
        super().__init__()

        self.token_embed = Embedding(vocab_size, dim)
        self.pos_embed   = PositionalEmbedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

        self.blocks = ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    attn_bias=attn_bias,
                    ffn_bias=ffn_bias,
                ) for _ in range(num_layers)
            ]
        )

        self._mask_cache: dict[int, Tensor] = {}

        self.ln_f = LayerNorm(dim)
        self.lm_head = Linear(dim, vocab_size, bias=False)

    def _get_causal_mask(self, seq_len: int) -> Tensor:
        if seq_len not in self._mask_cache:
            mask_np = np.triu(
                np.ones((seq_len, seq_len), dtype=np.float32),
                k=1
            ) * -1e9
            self._mask_cache[seq_len] = Tensor(mask_np, track_grad=False)
        return self._mask_cache[seq_len]

    def __call__(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        b, s = tokens.values.shape

        tok = self.token_embed(tokens)
        pos = self.pos_embed(s)
        x = tok + pos

        if mask is None:
            mask = self._get_causal_mask(s)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        tokenizer: Tokenizer,
        prompt: str,
        steps: int = 100,
        temperature: float = 1.0,
        top_k: int | None = 50
    ) -> str:

        ids = tokenizer.encode(prompt)
        ids = list(ids)

        self.set_trainability(False)

        for _ in range(steps):

            ctx = ids[-self.max_seq_len :]

            x = np.array(ctx, dtype=np.int32)[None, :]

            logits = self(Tensor(x))             
            last_logits = logits.values[0, -1]   

            scaled = last_logits / max(temperature, 1e-8)
            scaled -= np.max(scaled)

            probs = np.exp(scaled)
            probs /= np.sum(probs)

            if top_k is not None and top_k < len(probs):
                idx = np.argpartition(probs, -top_k)[-top_k:]
                top_probs = probs[idx]
                top_probs /= np.sum(top_probs)
                next_id = int(np.random.choice(idx, p=top_probs))
            else:
                next_id = int(np.random.choice(len(probs), p=probs))

            ids.append(next_id)

        return tokenizer.decode(ids)

