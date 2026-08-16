#!/usr/bin/env python
"""Generate a raw math-text pretraining corpus (no Question/Thought/Answer wrapper).

Goal: teach Jaiyu numeric/symbolic intuition before SFT reasoning format.
Every number is computed directly in Python (int/Fraction), never invented,
so nothing here can hallucinate.
"""

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

SEED = 26
TOTAL = 100_000
CATEGORY_FRACS = {
    "basic_facts": 0.30,
    "arithmetic": 0.28,
    "equations": 0.175,
    "fractions": 0.105,
    "prose": 0.07,
    "formulas": 0.07,
}

FRACTION_DENOMS = [2, 3, 4, 5, 6, 8, 10, 12]
# base denom -> multipliers k>1 such that base*k is also in FRACTION_DENOMS
EQUIV_MULTIPLIERS = {
    d: [k for k in range(2, 7) if d * k in FRACTION_DENOMS] for d in FRACTION_DENOMS
}

PROSE_FACTS = [
    "The sum of two numbers is their total.",
    "Subtraction is the inverse of addition.",
    "To add fractions, find a common denominator.",
    "Multiplication is repeated addition.",
    "Division is the inverse of multiplication.",
    "A fraction represents a part of a whole.",
    "Adding zero to a number leaves it unchanged.",
    "Multiplying a number by one leaves it unchanged.",
    "Multiplying a number by zero gives zero.",
    "The order of addends does not change the sum.",
    "The order of factors does not change the product.",
    "To subtract, take one number away from another.",
    "An equation states that two expressions are equal.",
    "Solving an equation means finding the value of the unknown.",
    "A variable is a symbol that stands for an unknown number.",
    "To simplify a fraction, divide the numerator and denominator by their greatest common factor.",
    "Two fractions are equivalent if they represent the same value.",
    "A coefficient is the number multiplied by a variable.",
    "The difference of two equal numbers is zero.",
    "Any number divided by one equals itself.",
    "The product of two negative numbers is positive.",
    "The product of a positive and a negative number is negative.",
    "A perimeter is the total distance around a shape.",
    "An area measures the space inside a two-dimensional shape.",
    "A rectangle has two pairs of equal, parallel sides.",
]


def _split_counts(total: int, fracs: dict) -> dict:
    counts = {k: int(total * f) for k, f in fracs.items()}
    counts[next(iter(fracs))] += total - sum(counts.values())
    return counts


def _decompose(n: int) -> str:
    tens, ones = (n // 10) * 10, n % 10
    return f"{n} = {tens} + {ones}"


def gen_arithmetic(rng: random.Random, n: int) -> list[str]:
    texts = []
    n_add, n_mul = n // 2, n - n // 2
    for _ in range(n_add):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        s = a + b
        assert a + b == s and s - b == a and s - a == b
        lines = [
            f"{a} + {b} = {s}",
            f"{b} + {a} = {s}",
            f"{s} - {b} = {a}",
            f"{s} - {a} = {b}",
            _decompose(a),
            _decompose(s),
        ]
        texts.append("\n".join(lines) + "\n")
    for _ in range(n_mul):
        p, q = rng.randint(1, 99), rng.randint(1, 99)
        r = p * q
        assert r // p == q and r // q == p
        lines = [
            f"{p} * {q} = {r}",
            f"{q} * {p} = {r}",
            f"{r} / {q} = {p}",
            f"{r} / {p} = {q}",
            _decompose(p),
            _decompose(r),
        ]
        texts.append("\n".join(lines) + "\n")
    return texts


def gen_basic_facts(rng: random.Random, n: int) -> list[str]:
    texts = []
    n_add = n // 4
    n_sub = n // 4
    n_mul = n // 4
    n_div = n - n_add - n_sub - n_mul
    for _ in range(n_add):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        s = a + b
        assert a + b == s
        texts.append(f"{a} + {b} = {s}\n")
    for _ in range(n_sub):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if a < b:
            a, b = b, a
        d = a - b
        assert a - b == d
        texts.append(f"{a} - {b} = {d}\n")
    for _ in range(n_mul):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        p = a * b
        assert a * b == p
        texts.append(f"{a} * {b} = {p}\n")
    for _ in range(n_div):
        d, q = rng.randint(1, 9), rng.randint(0, 9)
        dividend = d * q
        assert dividend // d == q
        texts.append(f"{dividend} / {d} = {q}\n")
    return texts


def gen_equations(rng: random.Random, n: int) -> list[str]:
    texts = []
    coeffs = [c for c in range(-10, 11) if c != 0]
    for _ in range(n):
        a = rng.choice(coeffs)
        b = rng.choice([c for c in range(-50, 51) if c != 0])
        x = rng.randint(-15, 15)
        c = a * x + b
        assert (c - b) // a == x and (c - b) % a == 0
        sign = "+" if b >= 0 else "-"
        lines = [
            f"{a}x {sign} {abs(b)} = {c}",
            f"{a}x = {c - b}",
            f"x = {x}",
        ]
        texts.append("\n".join(lines) + "\n")
    return texts


def _rand_fraction(rng: random.Random) -> Fraction:
    d = rng.choice(FRACTION_DENOMS)
    num = rng.randint(1, d - 1)
    return Fraction(num, d)


def gen_fractions(rng: random.Random, n: int) -> list[str]:
    texts = []
    equiv_bases = [d for d in FRACTION_DENOMS if EQUIV_MULTIPLIERS[d]]
    for _ in range(n):
        a1, b1 = _rand_fraction(rng), _rand_fraction(rng)
        a2, b2 = _rand_fraction(rng), _rand_fraction(rng)
        r1, r2 = a1 + b1, a2 + b2
        base = rng.choice(equiv_bases)
        k = rng.choice(EQUIV_MULTIPLIERS[base])
        num = rng.randint(1, base - 1)
        equiv = f"{num}/{base} = {num * k}/{base * k}"
        lines = [f"{a1} + {b1} = {r1}", equiv, f"{a2} + {b2} = {r2}"]
        texts.append("\n".join(lines) + "\n")
    return texts


def gen_prose(rng: random.Random, n: int) -> list[str]:
    texts = []
    for _ in range(n):
        k = rng.choice([1, 1, 2, 2, 3])
        facts = rng.sample(PROSE_FACTS, k) if k <= len(PROSE_FACTS) else PROSE_FACTS
        texts.append("\n".join(facts) + "\n")
    return texts


PI = 3.14


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(round(x, 2))


FORMULAS = [
    lambda rng: (("A", "pi * r * r"), {"r": rng.randint(1, 20)},
                 lambda v: PI * v["r"] * v["r"]),
    lambda rng: (("V", "l * w * h"), {"l": rng.randint(1, 20), "w": rng.randint(1, 20), "h": rng.randint(1, 20)},
                 lambda v: v["l"] * v["w"] * v["h"]),
    lambda rng: (("d", "r * t"), {"r": rng.randint(1, 20), "t": rng.randint(1, 20)},
                 lambda v: v["r"] * v["t"]),
    lambda rng: (("P", "2 * l + 2 * w"), {"l": rng.randint(1, 20), "w": rng.randint(1, 20)},
                 lambda v: 2 * v["l"] + 2 * v["w"]),
    lambda rng: (("C", "2 * pi * r"), {"r": rng.randint(1, 20)},
                 lambda v: 2 * PI * v["r"]),
    lambda rng: (("A", "l * w"), {"l": rng.randint(1, 20), "w": rng.randint(1, 20)},
                 lambda v: v["l"] * v["w"]),
    lambda rng: (("A", "0.5 * b * h"), {"b": rng.randint(1, 20), "h": rng.randint(1, 20)},
                 lambda v: 0.5 * v["b"] * v["h"]),
]


def gen_formulas(rng: random.Random, n: int) -> list[str]:
    texts = []
    for _ in range(n):
        (sym, pattern), values, compute = rng.choice(FORMULAS)(rng)
        result = compute(values)
        numeric = pattern
        for name, val in values.items():
            numeric = numeric.replace(name, _fmt(val))
        lines = [f"{sym} = {pattern}", f"{sym} = {numeric}", f"{sym} = {_fmt(result)}"]
        texts.append("\n".join(lines) + "\n")
    return texts


GENERATORS = {
    "basic_facts": gen_basic_facts,
    "arithmetic": gen_arithmetic,
    "equations": gen_equations,
    "fractions": gen_fractions,
    "prose": gen_prose,
    "formulas": gen_formulas,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pretrain/math_corpus.jsonl"))
    parser.add_argument("--num-examples", type=int, default=TOTAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    args = parser.parse_args()

    counts = _split_counts(args.num_examples, CATEGORY_FRACS)

    records = []
    for i, (category, count) in enumerate(counts.items()):
        rng = random.Random(args.seed + i)
        for text in GENERATORS[category](rng, count):
            records.append({"text": text, "category": category})

    random.Random(args.seed).shuffle(records)
    for i, rec in enumerate(records):
        rec["id"] = f"pretrain_{i}"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, recs: list[dict]) -> None:
        with path.open("w") as f:
            for rec in recs:
                f.write(json.dumps({"id": rec["id"], "text": rec["text"], "category": rec["category"]}) + "\n")

    if args.eval_ratio > 0:
        n_eval = int(len(records) * args.eval_ratio)
        train_recs, eval_recs = records[n_eval:], records[:n_eval]
        train_path = args.output.with_stem(args.output.stem + "_train")
        eval_path = args.output.with_stem(args.output.stem + "_eval")
        _write(train_path, train_recs)
        _write(eval_path, eval_recs)
        print(f"  train: {len(train_recs)} examples -> {train_path}")
        print(f"  eval: {len(eval_recs)} examples -> {eval_path}")
    else:
        _write(args.output, records)
        print("No eval split requested, generating single file.")
        print(f"  output: {args.output}")

    total_chars = sum(len(r["text"]) for r in records)
    print("Pretrain corpus generation complete:")
    print(f"  total examples: {len(records)}")
    for category, count in counts.items():
        print(f"  {category}: {count} examples")
    print(f"  estimated tokens: {total_chars // 4}")


if __name__ == "__main__":
    main()
