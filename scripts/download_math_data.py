#!/usr/bin/env python
"""Stream external math/language datasets into data/pretrain/external/*.jsonl."""

import argparse
import json
import re
import sys
from pathlib import Path

MATH_SYMBOLS = ("+", "-", "*", "/", "=", r"\frac", r"\sum", r"\int", r"\sqrt")

# Each entry: output name, candidate hub ids (first that loads wins), split,
# and a function turning one record into text (None -> skip record).
SOURCES = [
    (
        "openwebmath",
        ["open-web-math/open-web-math", "openwebmath"],
        "train",
        lambda r: r.get("text"),
        True,  # apply the math filter
    ),
    (
        "tinystories",
        ["roneneldan/TinyStories"],
        "train",
        lambda r: r.get("text"),
        False,
    ),
    (
        "math_qa",
        ["openai/gsm8k:main", "math_qa"],
        "train",
        lambda r: (
            f"{r['question']}\n{r['answer']}"
            if "question" in r and "answer" in r
            else (f"{r['Problem']}\n{r['Rationale']}" if "Problem" in r else None)
        ),
        False,
    ),
]

_DIGIT = re.compile(r"(\d)")
_SPACES = re.compile(r"[ \t]+")


def split_digits(text: str) -> str:
    """Space out every digit. Idempotent: repeated spaces are collapsed."""
    return _SPACES.sub(" ", _DIGIT.sub(r" \1 ", text))


def keep_math(text: str) -> bool:
    if len(text) < 100:
        return False
    if not any(sym in text for sym in MATH_SYMBOLS):
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) <= 0.30


def load_stream(candidates: list[str], split: str):
    from datasets import load_dataset

    for cand in candidates:
        name, _, config = cand.partition(":")
        try:
            return cand, load_dataset(name, config or None, split=split, streaming=True)
        except Exception as exc:  # dataset missing, gated, or offline
            print(f"  warning: could not load {cand}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return None, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-examples", type=int, default=1_000_000)
    p.add_argument("--output-dir", type=Path, default=Path("data/pretrain/external"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for out_name, candidates, split, to_text, math_filter in SOURCES:
        print(f"\n=== {out_name} ===")
        used, stream = load_stream(candidates, split)
        if stream is None:
            print(f"  skipped {out_name}: no candidate dataset available")
            continue

        out_path = args.output_dir / f"{out_name}.jsonl"
        kept = chars = 0
        with out_path.open("w", encoding="utf-8") as f:
            for record in stream:
                if kept >= args.max_examples:
                    break
                text = to_text(record)
                if not text:
                    continue
                if math_filter and not keep_math(text):
                    continue
                text = split_digits(text)
                f.write(json.dumps({
                    "id": f"{out_name}_{kept}",
                    "text": text,
                    "source": used,
                }) + "\n")
                kept += 1
                chars += len(text)

        # ~4 chars per token is the usual BPE rule of thumb; digit splitting
        # pushes it lower, so this is a deliberately rough upper estimate.
        print(f"  source: {used}")
        print(f"  entries: {kept}, chars: {chars}, est. tokens: ~{chars // 4}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
