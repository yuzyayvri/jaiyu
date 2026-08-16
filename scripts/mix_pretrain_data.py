#!/usr/bin/env python
"""Mix every pretraining source into one packed, tokenized 512-token corpus."""

import argparse
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SEQ_LEN = 512
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"

# name -> (share of total tokens, source jsonl or None for generated)
MIX = {
    "synthetic": (0.25, Path("data/pretrain/math_corpus.jsonl")),
    "openwebmath": (0.32, Path("data/pretrain/external/openwebmath.jsonl")),
    "math_qa": (0.03, Path("data/pretrain/external/math_qa.jsonl")),
    "calc": (0.15, None),
    "code": (0.10, None),
    "tinystories": (0.10, Path("data/pretrain/external/tinystories.jsonl")),
    "formulas": (0.05, None),
}

_DIGIT = re.compile(r"(\d)")
_SPACES = re.compile(r"[ \t]+")


def split_digits(text: str) -> str:
    """Space out every digit. Idempotent: repeated spaces are collapsed."""
    return _SPACES.sub(" ", _DIGIT.sub(r" \1 ", text))


# --------------------------------------------------------------------------
# generated sources; every number below is computed by Python, never written
# out by hand.
# --------------------------------------------------------------------------

def gen_calc(rng: random.Random) -> str:
    kind = rng.choice(["add", "sub", "mul", "div", "algebra", "fraction"])
    if kind == "add":
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        return f"{a} + {b} = <calc> {a} + {b} </calc> <result> {a + b} </result>"
    if kind == "sub":
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        return f"{a} - {b} = <calc> {a} - {b} </calc> <result> {a - b} </result>"
    if kind == "mul":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return f"{a} * {b} = <calc> {a} * {b} </calc> <result> {a * b} </result>"
    if kind == "div":
        b, q = rng.randint(2, 12), rng.randint(2, 12)
        a = b * q
        return f"{a} / {b} = <calc> {a} / {b} </calc> <result> {a // b} </result>"
    if kind == "algebra":
        a, x = rng.randint(2, 9), rng.randint(-9, 20)
        b = rng.randint(0, 99)
        c = a * x + b
        diff = c - b
        return (
            f"Solve {a}x + {b} = {c}: <calc> {c} - {b} </calc> <result> {diff} </result>, "
            f"then <calc> {diff} / {a} </calc> <result> {x} </result>. x = {x}"
        )
    n1, d1 = rng.randint(1, 9), rng.randint(2, 12)
    n2, d2 = rng.randint(1, 9), rng.randint(2, 12)
    res = Fraction(n1, d1) + Fraction(n2, d2)
    return (
        f"{n1}/{d1} + {n2}/{d2} = <calc> {n1}/{d1} + {n2}/{d2} </calc> "
        f"<result> {res} </result>"
    )


def gen_code(rng: random.Random) -> str:
    kind = rng.choice(["add", "sqrt", "fib"])
    if kind == "add":
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        return f"{a} + {b} = <code> print({a} + {b}) </code> <output> {a + b} </output>"
    if kind == "sqrt":
        n = rng.randint(2, 400)
        return (
            f"sqrt({n}) = <code> import math; print(math.sqrt({n})) </code> "
            f"<output> {math.sqrt(n)} </output>"
        )
    n = rng.randint(1, 30)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return (
        f"fib({n}) = <code> a,b=0,1\nfor _ in range({n}): a,b=b,a+b\nprint(a) </code> "
        f"<output> {a} </output>"
    )


def gen_formula(rng: random.Random) -> str:
    if rng.random() < 0.5:
        r = rng.randint(1, 20)
        return (
            f"A = pi * r * r. If r = {r}, A = <calc> 3.14159 * {r} * {r} </calc> "
            f"<result> {3.14159 * r * r} </result>"
        )
    l, w, h = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
    return (
        f"V = l * w * h. If l={l}, w={w}, h={h}, V = <calc> {l} * {w} * {h} </calc> "
        f"<result> {l * w * h} </result>"
    )


GENERATORS = {"calc": gen_calc, "code": gen_code, "formulas": gen_formula}


def file_docs(path: Path):
    """Yield texts from a jsonl file forever (reopened when exhausted)."""
    while True:
        with path.open(encoding="utf-8") as f:
            empty = True
            for line in f:
                if line.strip():
                    empty = False
                    yield json.loads(line)["text"]
        if empty:
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-tokens", type=int, default=1_000_000_000)
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--output", type=Path, default=Path("data/processed/pretrain_v2/train.npy"))
    p.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    p.add_argument("--smoke-test", action="store_true", help="Only produce 10000 tokens.")
    args = p.parse_args()

    total = 10_000 if args.smoke_test else args.total_tokens
    rng = random.Random(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab = tokenizer.get_vocab()
    eos_id, pad_id = vocab[EOS_TOKEN], vocab[PAD_TOKEN]
    unk_id = vocab.get("<unk>")

    # Build the live source table, dropping anything missing on disk.
    budgets, streams = {}, {}
    for name, (share, path) in MIX.items():
        if path is None:
            streams[name] = None
        elif path.exists():
            streams[name] = file_docs(path)
        else:
            raise SystemExit(
                f"missing {path} (category {name!r}, {100 * share:.0f}% of the mix). "
                f"Skipping it would silently produce a corpus with the wrong "
                f"proportions. Generated corpora are not in git; run "
                f"scripts/make_pretrain_data.py and scripts/download_math_data.py first."
            )
        budgets[name] = int(total * share)

    assert budgets, "no data sources available"

    n_chunks = max(1, total // SEQ_LEN)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.uint16, shape=(n_chunks, SEQ_LEN)
    )

    written = {name: 0 for name in budgets}
    unk_count = 0
    buf: list[int] = []
    cursor = 0  # chunks written so far

    # Interleaving by weighted random choice per document IS the shuffle: a
    # true shuffle of 1B tokens would not fit in RAM.
    # ponytail: document order within a source stays file order; only matters
    # if a source is internally sorted by difficulty.
    while cursor < n_chunks and budgets:
        names = list(budgets)
        weights = [MIX[n][0] for n in names]
        name = rng.choices(names, weights=weights)[0]

        gen = GENERATORS.get(name)
        if gen is not None:
            text = gen(rng)
        else:
            text = next(streams[name], None)
            if text is None:  # exhausted empty file
                del budgets[name]
                continue

        ids = tokenizer.encode(split_digits(text)).ids
        ids.append(eos_id)
        if unk_id is not None:
            unk_count += ids.count(unk_id)

        buf.extend(ids)
        written[name] += len(ids)
        if written[name] >= budgets[name]:
            del budgets[name]

        while len(buf) >= SEQ_LEN and cursor < n_chunks:
            out[cursor] = buf[:SEQ_LEN]
            del buf[:SEQ_LEN]
            cursor += 1

    if cursor < n_chunks:  # sources ran dry before the target
        buf.extend([pad_id] * (SEQ_LEN - len(buf) % SEQ_LEN))
        while len(buf) >= SEQ_LEN and cursor < n_chunks:
            out[cursor] = buf[:SEQ_LEN]
            del buf[:SEQ_LEN]
            cursor += 1
        out[cursor:] = pad_id

    out.flush()

    produced = sum(written.values())
    print(f"\nwrote {args.output}")
    for name, n in sorted(written.items(), key=lambda kv: -kv[1]):
        pct = 100 * n / produced if produced else 0
        print(f"  {name:<12} {n:>12,} tokens ({pct:4.1f}%, target {100*MIX[name][0]:.0f}%)")
    print(f"  {'TOTAL':<12} {produced:>12,} tokens")
    print(f"chunks: {n_chunks} x {SEQ_LEN}  shape {out.shape}")
    print(f"<unk> tokens: {unk_count}")
    print(f"file size: {out.nbytes / 1e9:.3f} GB")
    print("\nfirst chunk decoded:\n" + tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
