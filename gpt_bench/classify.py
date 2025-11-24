import numpy as np

from core.gpt import GPT
from core.tokenizer import CharTokenizer


SEQ_LEN       = 128
DIM           = 256
NUM_HEADS     = 4
NUM_LAYERS    = 4
FFN_EXPANSION = 4
ATTN_BIAS     = False
FFN_BIAS      = False

MODEL_PATH    = "gpt_model.npz"
DATA_PATH     = "data/input.txt"


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def classify(prompt: str, steps=200):
    text = load_text(DATA_PATH)
    tokenizer = CharTokenizer(text)

    model = GPT(
        vocab_size=len(tokenizer),
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
        ffn_expansion=FFN_EXPANSION,
        attn_bias=ATTN_BIAS,
        ffn_bias=FFN_BIAS,
    )

    model.load(MODEL_PATH)

    out = model.generate(
        tokenizer,
        prompt,
        steps=steps,
        temperature=0.8,
        top_k=40
    )

    print("\n=== Output ===\n")
    print(out)
    print()


if __name__ == "__main__":
    classify("Once upon a time")
