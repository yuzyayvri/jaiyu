#!/usr/bin/env python
"""Tokenize and pack one jsonl into the (N, 512) uint16 array train.py reads.

Used for held-out evaluation data. The training corpus goes through
mix_pretrain_data.py instead, which additionally interleaves several sources
at fixed proportions.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tokenizers import Tokenizer

from mix_pretrain_data import SEQ_LEN, EOS_TOKEN, PAD_TOKEN, split_digits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found.")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab = tokenizer.get_vocab()
    eos_id, pad_id = vocab[EOS_TOKEN], vocab[PAD_TOKEN]

    ids: list[int] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.extend(tokenizer.encode(split_digits(json.loads(line)["text"])).ids)
                ids.append(eos_id)

    if not ids:
        raise SystemExit(f"{args.input} produced no tokens.")

    # Pad the tail rather than dropping it; eval sets are small enough that a
    # discarded remainder would be a noticeable fraction.
    remainder = len(ids) % SEQ_LEN
    if remainder:
        ids.extend([pad_id] * (SEQ_LEN - remainder))

    arr = np.array(ids, dtype=np.uint16).reshape(-1, SEQ_LEN)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)

    print(f"wrote {args.output}")
    print(f"  {len(ids):,} tokens -> shape {arr.shape}")


if __name__ == "__main__":
    main()
