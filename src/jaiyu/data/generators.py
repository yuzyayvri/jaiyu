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


def generate_arithmetic(seed: int, n: int = 100) -> list[MathExample]:
    """Direct one-step arithmetic on operands in [0, 100].

    The reasoning step restates the question and gives the result, so every
    number in the Thought is derived from the Question. No place-value
    decomposition: the intermediate digits it invented were the main source of
    hallucinated numbers at this model size.
    """
    rng = random.Random(seed)
    ops = ["+", "-", "*", "/"]
    examples = []
    for i in range(n):
        op = rng.choice(ops)

        if op == "+":
            a, b = rng.randint(0, 100), rng.randint(0, 100)
            result = a + b
            thought = f"Calculate {a} + {b}. The sum is {result}."
            difficulty = 1
        elif op == "-":
            a = rng.randint(0, 100)
            b = rng.randint(0, a)  # b <= a keeps the difference non-negative
            result = a - b
            thought = f"Calculate {a} - {b}. The difference is {result}."
            difficulty = 1
        elif op == "*":
            a, b = rng.randint(0, 100), rng.randint(0, 100)
            result = a * b
            thought = f"Multiply {a} by {b}. The product is {result}."
            difficulty = 2
        else:
            b = rng.randint(1, 100)
            a = b * rng.randint(0, 100 // b)  # exact division, a <= 100
            result = a // b
            thought = f"Divide {a} by {b}. The quotient is {result}."
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
        a = rng.randint(1, 10)
        b = _nonzero(rng, -50, 50)
        x = rng.randint(-15, 15)
        c = a * x + b  # so (c - b) / a == x exactly

        lhs = f"{a}x {'+' if b >= 0 else '-'} {abs(b)}"
        # Every number below comes straight from the question: the coefficient
        # `a` is repeated in both steps so it cannot drift mid-solution.
        thought = (
            f"Solve for x. "
            f"Move {b} to the other side: {a}x = {c - b}. "
            f"Divide both sides by {a}: x = {x}."
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
