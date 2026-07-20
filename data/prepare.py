from pathlib import Path

import numpy as np
from tokenizer import SAVE_PATH as TOKENIZER_PATH
from tokenizer import encode, load_tokenizer

CORPUS_PATH = Path(__file__).parent / "raw" / "corpus.txt"
OUT_DIR = Path(__file__).parent
TRAIN_BIN = OUT_DIR / "train.bin"
VAL_BIN = OUT_DIR / "val.bin"
VAL_FRACTION = 0.1  # last 10% of the corpus held out for validation

def prepare(
    corpus_path: Path = CORPUS_PATH,
    tokenizer_path: Path = TOKENIZER_PATH,
    val_fraction: float = VAL_FRACTION,
) -> None:
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"No tokenizer found at {tokenizer_path}. Run tokenizer.py first to train one."
        )
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus found at {corpus_path}.")

    # Load tokenizer
    tok = load_tokenizer(tokenizer_path)

    # Load corpus file
    text = corpus_path.read_text(encoding="utf-8")

    # encode
    ids = encode(tok, text)

    # split into train/val by position
    split_idx = int(len(ids) * (1 - val_fraction))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    vocab_size = tok.get_vocab_size()

    # uint16 covers vocab sizes up to 65536 -- fine for 8k vocab.
    # switch to uint32 if you ever grow vocab_size past that.
    dtype = np.uint16 if vocab_size < 65536 else np.uint32

    train_arr = np.array(train_ids, dtype=dtype)
    val_arr = np.array(val_ids, dtype=dtype)

    train_arr.tofile(TRAIN_BIN)
    val_arr.tofile(VAL_BIN)

    print(f"Train: {len(train_arr):,} tokens -> {TRAIN_BIN}")
    print(f"Val:   {len(val_arr):,} tokens -> {VAL_BIN}")

if __name__ == "__main__":
    prepare()