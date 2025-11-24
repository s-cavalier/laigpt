import os
import numpy as np
from tqdm import tqdm

from autograd.tensor import Tensor
from autograd.functions import CrossEntropyLoss
from core.datastream import TextSource, GPTData
from core.gpt import GPT
from core.optimize import Adam
from core.tokenizer import CharTokenizer


DATA_PATH       = "gpt_bench/input.txt"
SAVE_PATH       = "gpt_model.npz"

SEQ_LEN         = 128
BATCH_SIZE      = 32
EPOCHS          = 5
LEARNING_RATE   = 3e-4
VAL_FRACTION    = 0.1

DIM             = 256
NUM_HEADS       = 4
NUM_LAYERS      = 4
FFN_EXPANSION   = 4
ATTN_BIAS       = False
FFN_BIAS        = False

def load_text(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    np.random.seed(42)

    text = load_text(DATA_PATH)
    print(f"Loaded dataset with {len(text)} characters.")

    tokenizer = CharTokenizer(text)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    ids = np.array(tokenizer.encode(text), dtype=np.int32)
    n = len(ids)
    print(f"Total tokens: {n}")

    split = int(n * (1 - VAL_FRACTION))
    train_ids = ids[:split]
    val_ids   = ids[split:]

    train_src = TextSource(train_ids)
    val_src   = TextSource(val_ids)

    train_stream = GPTData(train_src, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True)
    val_stream   = GPTData(val_src,   seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False)

    model = GPT(
        vocab_size=vocab_size,
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
        ffn_expansion=FFN_EXPANSION,
        attn_bias=ATTN_BIAS,
        ffn_bias=FFN_BIAS,
    )

    optimizer = Adam(model, lr=LEARNING_RATE)
    
    target_buf = np.zeros(BATCH_SIZE * SEQ_LEN, dtype=np.int32)
    loss_fn = CrossEntropyLoss(target_buf)

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        model.set_trainability(True)

        train_iter = iter(train_stream)
        train_losses = []

        pbar = tqdm(train_iter, total=len(train_stream), desc="Training")

        for X_np, Y_np in pbar:

            X = Tensor(X_np)

            target_buf[:] = Y_np.reshape(-1)

            logits = model(X)

            b, s, v = logits.values.shape
            logits_flat = logits.reshape((b * s, v))

            loss = loss_fn(logits_flat)

            optimizer.zero_gradients()
            loss.backpropagate()

            optimizer.step()

            train_losses.append(loss.values)
            if len(train_losses) % 50 == 0:
                pbar.set_postfix({"loss": float(np.mean(train_losses[-50:]))})

        print(f"Train loss: {float(np.mean(train_losses)):.4f}")

        model.set_trainability(False)
        val_losses = []
        val_iter = iter(val_stream)

        for X_np, Y_np in val_iter:
            X = Tensor(X_np)
            targets = Y_np.reshape(-1)

            logits = model(X)
            b, s, v = logits.values.shape
            logits_flat = logits.reshape((b * s, v))

            loss_fn = CrossEntropyLoss(targets)
            loss = loss_fn(logits_flat)

            val_losses.append(loss.values)

        if val_losses:
            print(f"Val loss: {float(np.mean(val_losses)):.4f}")

        model.save(f"gpt_epoch_{epoch+1}.npz")

    model.save(SAVE_PATH)
    print(f"\nTraining complete. Saved final model to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
