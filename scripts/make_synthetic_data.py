#!/usr/bin/env python
"""Generate synthetic, verifiable math problems for training/eval.

Writes train.jsonl and eval.jsonl of MathExample records, split roughly
evenly across arithmetic, fractions, and linear equations.
"""

import argparse
import dataclasses
import json
import random
from pathlib import Path

from jaiyu.data.generators import (
    MathExample,
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


def _replay_examples(path: Path, fraction: float, n_train: int,
                     rng: random.Random) -> list[MathExample]:
    """Pre-training text carried into fine-tuning to limit forgetting.

    Fine-tuning only on question-and-answer maths pulls the model away from the
    prose, code and formula text it spent a billion tokens learning. A slice of
    the original corpus keeps that alive; it has no question or answer, so it
    is stored as bare text the packer emits unchanged.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found; generate it with scripts/make_pretrain_data.py.")

    lines = [json.loads(line)["text"] for line in path.open() if line.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty.")

    n_replay = int(n_train * fraction / max(1e-9, 1 - fraction))
    chosen = [rng.choice(lines) for _ in range(n_replay)]
    return [
        MathExample(
            id=f"replay_{i}",
            text=text if text.endswith("\n") else text + "\n",
            topic="replay",
            difficulty=rng.randint(1, 4),
            reasoning=[],
            source="pretrain_replay",
            split="train",
            answer="",
        )
        for i, text in enumerate(chosen)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/intermediate/synthetic"))
    # The model this data fine-tunes has seen a billion tokens; ten thousand
    # examples is enough to teach the answer format and not enough to teach a
    # procedure it does not already have.
    parser.add_argument("--num-train", type=int, default=60000)
    parser.add_argument("--num-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--curriculum", action="store_true",
                        help="Order training examples easiest-first by difficulty "
                             "instead of shuffling, so early steps see foundations.")
    parser.add_argument("--replay", type=Path, default=None,
                        help="Pre-training corpus (jsonl) to mix in, keeping the prose "
                             "and code fluency that fine-tuning on maths alone erodes.")
    parser.add_argument("--replay-fraction", type=float, default=0.15,
                        help="Share of the training set drawn from --replay.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    topics = list(GENERATORS)
    train_counts = _split_counts(args.num_train, len(topics))
    eval_counts = _split_counts(args.num_eval, len(topics))

    summary: dict[str, int] = {}
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"

    rng = random.Random(args.seed)
    train_examples: list = []
    eval_examples: list = []

    for i, topic in enumerate(topics):
        n_train, n_eval = train_counts[i], eval_counts[i]
        examples = GENERATORS[topic](seed=args.seed + i, n=n_train + n_eval)
        rng.shuffle(examples)

        for ex, split in zip(examples, ["train"] * n_train + ["eval"] * n_eval):
            ex.split = split
            (train_examples if split == "train" else eval_examples).append(ex)

        summary[topic] = n_train + n_eval

    # Repetition within train is fine; only eval must stay disjoint from train.
    train_texts = {ex.text for ex in train_examples}
    eval_examples = [ex for ex in eval_examples if ex.text not in train_texts]

    if args.replay:
        train_examples.extend(_replay_examples(args.replay, args.replay_fraction,
                                               len(train_examples), rng))

    if args.curriculum:
        # Stable sort: difficulty decides the order, and examples of equal
        # difficulty keep the shuffle they already have.
        train_examples.sort(key=lambda ex: ex.difficulty)
    else:
        rng.shuffle(train_examples)

    with train_path.open("w") as train_f, eval_path.open("w") as eval_f:
        for ex in train_examples:
            train_f.write(json.dumps(dataclasses.asdict(ex)) + "\n")
        for ex in eval_examples:
            eval_f.write(json.dumps(dataclasses.asdict(ex)) + "\n")

    print("Synthetic data generation complete:")
    for topic, count in summary.items():
        print(f"  {topic}: {count} examples")
    print(f"  total train: {len(train_examples)} -> {train_path}")
    print(f"  total eval:  {len(eval_examples)} (of {args.num_eval} requested, "
          f"dupes-of-train dropped) -> {eval_path}")


if __name__ == "__main__":
    main()
