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
    """Direct arithmetic with a place-value trace for +/- and a one-step
    trace for */÷, so every number in the Thought is derived from the
    Question (no invented intermediates).
    """
    rng = random.Random(seed)
    ops = ["+", "-", "*", "/"]
    examples = []
    for i in range(n):
        op = rng.choice(ops)

        if op == "+":
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            result = a + b
            a_tens, a_ones = divmod(a, 10)
            b_tens, b_ones = divmod(b, 10)
            raw_ones_sum = a_ones + b_ones
            carry = 1 if raw_ones_sum >= 10 else 0
            ones_sum = raw_ones_sum - 10 * carry
            tens_sum = a_tens + b_tens + carry

            if carry:
                thought = (
                    f"Add the ones: {a_ones} + {b_ones} = {raw_ones_sum}. "
                    f"Write down {ones_sum}, carry {carry} to the tens. "
                    f"Add the tens: {a_tens} + {b_tens} + {carry} (carry) = {tens_sum}. "
                    f"Combine: {tens_sum * 10} + {ones_sum} = {result}."
                )
            else:
                thought = (
                    f"Add the ones: {a_ones} + {b_ones} = {ones_sum}. "
                    f"Add the tens: {a_tens} + {b_tens} = {tens_sum}. "
                    f"Combine: {tens_sum * 10} + {ones_sum} = {result}."
                )
            difficulty = 1
        elif op == "-":
            a = rng.randint(10, 99)
            b = rng.randint(10, a)  # b <= a keeps the difference non-negative
            result = a - b
            a_tens, a_ones = divmod(a, 10)
            b_tens, b_ones = divmod(b, 10)

            if a_ones < b_ones:
                borrowed_tens = a_tens - 1
                borrowed_ones = a_ones + 10
                ones_diff = borrowed_ones - b_ones
                tens_diff = borrowed_tens - b_tens
                thought = (
                    f"Subtract the ones: {a_ones} - {b_ones} is not possible. "
                    f"Borrow 1 from the tens, making the {a_tens} a {borrowed_tens}, "
                    f"and the {a_ones} a {borrowed_ones}. "
                    f"{borrowed_ones} - {b_ones} = {ones_diff}. "
                    f"Subtract the tens: {borrowed_tens} - {b_tens} = {tens_diff}. "
                    f"Combine: {tens_diff * 10} + {ones_diff} = {result}."
                )
            else:
                ones_diff = a_ones - b_ones
                tens_diff = a_tens - b_tens
                thought = (
                    f"Subtract the ones: {a_ones} - {b_ones} = {ones_diff}. "
                    f"Subtract the tens: {a_tens} - {b_tens} = {tens_diff}. "
                    f"Combine: {tens_diff * 10} + {ones_diff} = {result}."
                )
            difficulty = 1
        elif op == "*":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            result = a * b
            thought = f"Multiply {a} by {b}. The product is {result}."
            difficulty = 2
        else:
            b = rng.randint(2, 12)
            a = b * rng.randint(2, 12)
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
