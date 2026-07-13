from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
CORPUS_PATH = Path(__file__).parent / "raw" / "corpus.txt" 
SAVE_PATH = Path(__file__).parent / "yapuny_tokenizer.json"


def train_tokenizer(corpus_path: Path = CORPUS_PATH, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the given corpus file."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path = SAVE_PATH) -> None:
    tokenizer.save(str(path))
    print(f"Saved tokenizer to {path}")


def load_tokenizer(path: Path = SAVE_PATH) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def round_trip_test(tokenizer: Tokenizer, text: str) -> bool:
    """Sanity check: encode then decode should return (close to) the original."""
    ids = encode(tokenizer, text)
    decoded = decode(tokenizer, ids)
    print(f"Original : {text!r}")
    print(f"Token IDs: {ids}")
    print(f"Decoded  : {decoded!r}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    return decoded.strip() == text.strip()


if __name__ == "__main__":
    if not CORPUS_PATH.exists():
        print(f"No corpus found at {CORPUS_PATH}.")
        print("Create data/raw/corpus.txt with your training text, then rerun.")
    else:
        tok = train_tokenizer()
        save_tokenizer(tok)
        ok = round_trip_test(tok, "Testing Yapuny's tokenizer, let's see how it splits!")
        print("Round-trip OK" if ok else "Round-trip MISMATCH -- investigate")