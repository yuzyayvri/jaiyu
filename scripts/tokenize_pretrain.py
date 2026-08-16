"""Tokenize and pack the raw pretraining math corpus into fixed-length chunks."""
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SEQ_LEN = 512
EOS_ID = 1
PAD_ID = 0

IN_PATH = Path("data/pretrain/math_corpus.jsonl")
OUT_PATH = Path("data/processed/pretrain/train.npy")


def load_examples(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_and_pack(examples: list[dict], tokenizer: Tokenizer) -> np.ndarray:
    ids: list[int] = []
    for ex in examples:
        ids.extend(tokenizer.encode(ex["text"]).ids)
        ids.append(EOS_ID)

    n_chunks = (len(ids) + SEQ_LEN - 1) // SEQ_LEN
    padded_len = n_chunks * SEQ_LEN
    ids.extend([PAD_ID] * (padded_len - len(ids)))

    return np.array(ids, dtype=np.uint16).reshape(n_chunks, SEQ_LEN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", type=Path, default=IN_PATH)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    parser.add_argument("--tokenizer", default="data/tokenizer/jaiyu_tokenizer.json")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    examples = load_examples(args.in_path)
    arr = tokenize_and_pack(examples, tokenizer)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, arr)

    print(f"pretrain: {len(examples)} examples, {arr.size} tokens, shape {arr.shape} -> {args.out_path}")


if __name__ == "__main__":
    main()
