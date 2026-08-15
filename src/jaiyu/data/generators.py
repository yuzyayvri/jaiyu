"""Data generation, tokenization, and loading utilities.

TODO: tokenizer training/loading helpers.
TODO: dataset/dataloader classes for train and held-out eval splits.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class MathExample:
    id: str
    text: str
    topic: str
    difficulty: int
    reasoning: list[str]
    source: str
    split: str
    answer: str


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    while True:
        n = rng.randint(lo, hi)
        if n != 0:
            return n


_PLACE_NAMES = ["ones", "tens", "hundreds", "thousands"]


def _place_digits(n: int) -> list[int]:
    """Digits of n from least to most significant, one per entry in _PLACE_NAMES."""
    digits = []
    for p in range(len(_PLACE_NAMES)):
        digits.append((n // (10**p)) % 10)
    return digits


def _addition_thought(a: int, b: int, result: int) -> str:
    da, db = _place_digits(a), _place_digits(b)
    parts = []
    for i, place in enumerate(_PLACE_NAMES):
        mult = 10**i
        av, bv = da[i] * mult, db[i] * mult
        if av == 0 and bv == 0:
            continue
        parts.append(f"Add the {place}: {av} + {bv} = {av + bv}.")
    combine = " + ".join(str(da[i] * 10**i + db[i] * 10**i) for i in range(len(_PLACE_NAMES)))
    parts.append(f"Combine: {combine} = {result}.")
    return " ".join(parts)


def _subtraction_thought(a: int, b: int, result: int) -> str:
    """Assumes a >= b, so no negative intermediate results."""
    da, db = _place_digits(a), _place_digits(b)
    borrow_in = 0
    diffs = []
    parts = []
    for i, place in enumerate(_PLACE_NAMES):
        mult = 10**i
        ai, bi = da[i] - borrow_in, db[i]
        borrowed = ai < bi
        if borrowed:
            ai += 10
        d = ai - bi
        diffs.append(d * mult)
        note = " (after borrowing 1 from the next place)" if borrowed else ""
        parts.append(f"Subtract the {place}: {ai} - {bi} = {d}{note}.")
        borrow_in = 1 if borrowed else 0
    combine = " + ".join(str(v) for v in reversed(diffs)) or "0"
    parts.append(f"Combine: {combine} = {result}.")
    return " ".join(parts)


def generate_arithmetic(seed: int, n: int = 100) -> list[MathExample]:
    rng = random.Random(seed)
    ops = ["+", "-", "*", "/"]
    examples = []
    for i in range(n):
        op = rng.choice(ops)

        if op in ("+", "-"):
            a = rng.randint(0, 9999)
            if op == "+":
                b = rng.randint(0, 9999)
                result = a + b
                thought = _addition_thought(a, b, result)
            else:
                # b <= a keeps the difference non-negative.
                b = rng.randint(0, a)
                result = a - b
                thought = _subtraction_thought(a, b, result)
            difficulty = 1
        elif op == "*":
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            result = a * b
            thought = f"Multiply {a} by {b}: {a} * {b} = {result}."
            difficulty = 2
        else:
            b = rng.randint(1, 999)
            a = b * rng.randint(0, 999)  # ensure exact division
            result = a // b
            thought = f"Divide {a} by {b}: {a} / {b} = {result}."
            difficulty = 2

        question = f"What is {a} {op} {b}?"
        text = f"Question: {question}\nThought: {thought}\n"
        examples.append(
            MathExample(
                id=f"arithmetic_{seed}_{i}",
                text=text,
                topic="arithmetic",
                difficulty=difficulty,
                reasoning=["calculation"],
                source="synthetic_arithmetic",
                split="train",
                answer=str(result),
            )
        )
    return examples


def generate_fractions(seed: int, n: int = 100) -> list[MathExample]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        a = Fraction(_nonzero(rng, -9, 9), rng.randint(2, 9))
        b = Fraction(_nonzero(rng, -9, 9), rng.randint(2, 9))
        op = rng.choice(["+", "*"])

        if op == "+":
            result = a + b
            L = math.lcm(a.denominator, b.denominator)
            ca = a.numerator * (L // a.denominator)
            cb = b.numerator * (L // b.denominator)
            thought = (
                f"Find a common denominator, which is {L}. "
                f"Convert {a} to {ca}/{L}. Convert {b} to {cb}/{L}. "
                f"Now add: {ca}/{L} + {cb}/{L} = {result}."
            )
        else:
            result = a * b
            thought = (
                f"Multiply the numerators and multiply the denominators: "
                f"{a} * {b} = {result}."
            )

        question = f"What is {a} {op} {b}? Give the answer as a reduced fraction."
        text = f"Question: {question}\nThought: {thought}\n"
        examples.append(
            MathExample(
                id=f"fractions_{seed}_{i}",
                text=text,
                topic="fractions",
                difficulty=3,
                reasoning=["calculation", "transformation"],
                source="synthetic_fractions",
                split="train",
                answer=str(result),
            )
        )
    return examples


def generate_linear_equations(seed: int, n: int = 100) -> list[MathExample]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        a = rng.randint(1, 50)
        b = rng.randint(-1000, 1000)
        x = rng.randint(-15, 15)
        c = a * x + b

        lhs_sign = "+" if b >= 0 else "-"
        lhs = f"{a}x {lhs_sign} {abs(b)}"
        if b >= 0:
            verb, prep, amt, side_op = "subtract", "from", b, "-"
        else:
            verb, prep, amt, side_op = "add", "to", -b, "+"
        new_c = c - b

        thought = (
            f"First, {verb} {amt} {prep} both sides. "
            f"{lhs} {side_op} {amt} = {c} {side_op} {amt}. "
            f"This simplifies to {a}x = {new_c}. "
            f"Next, divide both sides by {a}. "
            f"{a}x / {a} = {new_c} / {a}. "
            f"This gives x = {x}."
        )
        question = f"Solve {lhs} = {c}."
        text = f"Question: {question}\nThought: {thought}\n"
        examples.append(
            MathExample(
                id=f"linear_{seed}_{i}",
                text=text,
                topic="algebra",
                difficulty=4,
                reasoning=["transformation", "substitution"],
                source="synthetic_linear_equations",
                split="train",
                answer=f"x = {x}",
            )
        )
    return examples


def make_synthetic_examples(topic: str, difficulty: int, n: int):
    raise NotImplementedError("TODO: implement synthetic data generation")


def load_tokenizer(path: str):
    raise NotImplementedError("TODO: implement tokenizer loading")
