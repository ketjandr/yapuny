"""
Yapuny single entry point.

Usage:
    python run_pipeline.py              # run all steps
    python run_pipeline.py tokenize     # only train tokenizer
    python run_pipeline.py prepare      # only prepare data
    python run_pipeline.py train        # only train model
"""

import sys


def step_tokenize():
    from data.tokenizer import (
        CORPUS_PATH,
        round_trip_test,
        save_tokenizer,
        train_tokenizer,
    )

    if not CORPUS_PATH.exists():
        print(f"No corpus at {CORPUS_PATH}. Add your text there first.")
        sys.exit(1)

    tok = train_tokenizer()
    save_tokenizer(tok)
    round_trip_test(tok, "Testing Yapuny's tokenizer, let's see how it splits!")

def step_prepare():
    from data.prepare import prepare
    prepare()

def step_train():
    from train import main
    main()

STEPS = {
    "tokenize": step_tokenize,
    "prepare": step_prepare,
    "train": step_train,
}

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else STEPS.keys()

    for name in requested:
        if name not in STEPS:
            print(f"Enter a valid command. Choose from {', '.join(STEPS.keys())}")
            sys.exit(1)
        STEPS[name]()