"""Tokenize and pack Jaiyu synthetic math data into fixed-length chunks."""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SEQ_LEN = 512
EOS_ID = 50256

IN_DIR = Path("data/intermediate/synthetic")
OUT_DIR = Path("data/processed/tokenized")


def load_examples(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_and_pack(examples: list[dict], tokenizer: Tokenizer) -> np.ndarray:
    ids: list[int] = []
    for ex in examples:
        full_text = ex["text"] + "Answer: " + ex["answer"] + "\n"
        ids.extend(tokenizer.encode(full_text).ids)

    n_chunks = (len(ids) + SEQ_LEN - 1) // SEQ_LEN
    padded_len = n_chunks * SEQ_LEN
    ids.extend([EOS_ID] * (padded_len - len(ids)))

    arr = np.array(ids, dtype=np.uint16).reshape(n_chunks, SEQ_LEN)
    return arr


def process(split: str, tokenizer: Tokenizer) -> None:
    examples = load_examples(IN_DIR / f"{split}.jsonl")
    arr = tokenize_and_pack(examples, tokenizer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.npy"
    np.save(out_path, arr)

    print(f"{split}: {arr.size} tokens, shape {arr.shape} -> {out_path}")


def main() -> None:
    tokenizer = Tokenizer.from_pretrained("gpt2")
    process("train", tokenizer)
    process("eval", tokenizer)


if __name__ == "__main__":
    main()
