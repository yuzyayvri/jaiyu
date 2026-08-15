"""Data generation, tokenization, and loading utilities.

TODO: tokenizer training/loading helpers.
TODO: dataset/dataloader classes for train and held-out eval splits.
"""

from __future__ import annotations

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


def _split(rng: random.Random) -> str:
    return "eval" if rng.random() < 0.1 else "train"


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    while True:
        n = rng.randint(lo, hi)
        if n != 0:
            return n


def generate_arithmetic(seed: int, n: int = 100) -> list[MathExample]:
    rng = random.Random(seed)
    ops = ["+", "-", "*", "/"]
    examples = []
    for i in range(n):
        op = rng.choice(ops)
        a = rng.randint(-50, 50)
        if op == "/":
            b = _nonzero(rng, -12, 12)
            a = b * rng.randint(-12, 12)  # ensure exact division
        else:
            b = rng.randint(-50, 50)

        if op == "+":
            result = a + b
            thought = f"Add {a} and {b}: {a} + {b} = {result}."
            difficulty = 1
        elif op == "-":
            result = a - b
            thought = f"Subtract {b} from {a}: {a} - {b} = {result}."
            difficulty = 1
        elif op == "*":
            result = a * b
            thought = f"Multiply {a} by {b}: {a} * {b} = {result}."
            difficulty = 2
        else:
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
                split=_split(rng),
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
            thought = (
                f"Find a common denominator for {a} and {b}, then add the "
                f"numerators: {a} + {b} = {result}."
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
                split=_split(rng),
                answer=str(result),
            )
        )
    return examples


def generate_linear_equations(seed: int, n: int = 100) -> list[MathExample]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        a = _nonzero(rng, -9, 9)
        b = rng.randint(-20, 20)
        x = rng.randint(-15, 15)
        c = a * x + b

        thought = (
            f"Subtract {b} from both sides: {a}x = {c} - {b} = {c - b}. "
            f"Divide both sides by {a}: x = {c - b} / {a} = {x}."
        )
        question = f"Solve for x: {a}x + {b} = {c}"
        text = f"Question: {question}\nThought: {thought}\n"
        examples.append(
            MathExample(
                id=f"linear_{seed}_{i}",
                text=text,
                topic="algebra",
                difficulty=4,
                reasoning=["transformation", "substitution"],
                source="synthetic_linear_equations",
                split=_split(rng),
                answer=str(x),
            )
        )
    return examples


def make_synthetic_examples(topic: str, difficulty: int, n: int):
    raise NotImplementedError("TODO: implement synthetic data generation")


def load_tokenizer(path: str):
    raise NotImplementedError("TODO: implement tokenizer loading")
