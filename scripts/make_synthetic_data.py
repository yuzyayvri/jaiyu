#!/usr/bin/env python
"""Generate synthetic, verifiable math problems for training/eval.

Writes train.jsonl and eval.jsonl of MathExample records, split roughly
evenly across arithmetic, fractions, and linear equations.
"""

import argparse
import dataclasses
import json
from pathlib import Path

from jaiyu.data.generators import (
    generate_arithmetic,
    generate_fractions,
    generate_linear_equations,
)

GENERATORS = {
    "arithmetic": generate_arithmetic,
    "fractions": generate_fractions,
    "linear_equations": generate_linear_equations,
}


def _split_counts(total: int, n_groups: int) -> list[int]:
    base, remainder = divmod(total, n_groups)
    return [base + (1 if i < remainder else 0) for i in range(n_groups)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/intermediate/synthetic"))
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    topics = list(GENERATORS)
    train_counts = _split_counts(args.num_train, len(topics))
    eval_counts = _split_counts(args.num_eval, len(topics))

    summary: dict[str, int] = {}
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"

    with train_path.open("w") as train_f, eval_path.open("w") as eval_f:
        for i, topic in enumerate(topics):
            n_train, n_eval = train_counts[i], eval_counts[i]
            examples = GENERATORS[topic](seed=args.seed + i, n=n_train + n_eval)

            for ex, split in zip(examples, ["train"] * n_train + ["eval"] * n_eval):
                ex.split = split
                out = train_f if split == "train" else eval_f
                out.write(json.dumps(dataclasses.asdict(ex)) + "\n")

            summary[topic] = n_train + n_eval

    print("Synthetic data generation complete:")
    for topic, count in summary.items():
        print(f"  {topic}: {count} examples")
    print(f"  total train: {args.num_train} -> {train_path}")
    print(f"  total eval:  {args.num_eval} -> {eval_path}")


if __name__ == "__main__":
    main()
