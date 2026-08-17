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


# Fraction of examples that skip the calculator entirely, so the model keeps
# some internal arithmetic instead of offloading every single step.
PURE_REASONING_RATE = 0.1


def _calc(expr: str, value) -> str:
    """Tool-call span matching the pre-training corpus format."""
    return f"<calc> {expr} </calc> <result> {value} </result>"


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

        if rng.random() < PURE_REASONING_RATE:
            # Pure reasoning: single digits, no calculator.
            op = rng.choice(["+", "-"])
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            if op == "-":
                a, b = max(a, b), min(a, b)
            result = a + b if op == "+" else a - b
            verb = "Add" if op == "+" else "Subtract"
            thought = f"{verb} the single digits: {a} {op} {b} = {result}."
            difficulty = 1
        elif op == "+":
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
                    f"Add the ones: {_calc(f'{a_ones} + {b_ones}', raw_ones_sum)}. "
                    f"Write down {ones_sum}, carry {carry} to the tens. "
                    "Add the tens: "
                    f"{_calc(f'{a_tens} + {b_tens} + {carry}', tens_sum)}. "
                    f"Combine: {_calc(f'{tens_sum * 10} + {ones_sum}', result)}."
                )
            else:
                thought = (
                    f"Add the ones: {_calc(f'{a_ones} + {b_ones}', ones_sum)}. "
                    f"Add the tens: {_calc(f'{a_tens} + {b_tens}', tens_sum)}. "
                    f"Combine: {_calc(f'{tens_sum * 10} + {ones_sum}', result)}."
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
                    f"{_calc(f'{borrowed_ones} - {b_ones}', ones_diff)}. "
                    "Subtract the tens: "
                    f"{_calc(f'{borrowed_tens} - {b_tens}', tens_diff)}. "
                    f"Combine: {_calc(f'{tens_diff * 10} + {ones_diff}', result)}."
                )
            else:
                ones_diff = a_ones - b_ones
                tens_diff = a_tens - b_tens
                thought = (
                    f"Subtract the ones: {_calc(f'{a_ones} - {b_ones}', ones_diff)}. "
                    f"Subtract the tens: {_calc(f'{a_tens} - {b_tens}', tens_diff)}. "
                    f"Combine: {_calc(f'{tens_diff * 10} + {ones_diff}', result)}."
                )
            difficulty = 1
        elif op == "*":
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            result = a * b
            thought = f"Multiply: {_calc(f'{a} * {b}', result)}."
            difficulty = 2
        else:
            b = rng.randint(2, 12)
            a = b * rng.randint(2, 12)
            result = a // b
            thought = f"Divide: {_calc(f'{a} / {b}', result)}."
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

        if rng.random() < PURE_REASONING_RATE:
            # Pure reasoning: same denominator, single-digit numerators, no calculator.
            op = "+"
            d = rng.randint(2, 9)
            # Coprime numerators keep the printed forms as /d, so the "denominators
            # already match" narration stays true.
            na = rng.choice([k for k in range(1, 10) if math.gcd(k, d) == 1])
            nb = rng.choice([k for k in range(1, 10) if math.gcd(k, d) == 1])
            a, b = Fraction(na, d), Fraction(nb, d)
            result = a + b
            thought = (
                f"The denominators already match, so add the numerators: "
                f"{a.numerator} + {b.numerator} = {a.numerator + b.numerator}, "
                f"giving {result}."
            )
        elif op == "+":
            result = a + b
            L = math.lcm(a.denominator, b.denominator)
            ca = a.numerator * (L // a.denominator)
            cb = b.numerator * (L // b.denominator)
            thought = (
                f"Common denominator is {L}. "
                f"Convert: {a} = {ca}/{L}. Convert: {b} = {cb}/{L}. "
                f"Add: {_calc(f'{ca}/{L} + {cb}/{L}', result)}."
            )
        else:
            result = a * b
            thought = (
                f"Multiply the numerators and multiply the denominators: "
                f"{_calc(f'{a} * {b}', result)}."
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

        if rng.random() < PURE_REASONING_RATE:
            # Pure reasoning: x + b = c with single-digit numbers, no calculator.
            a = 1
            b = rng.randint(1, 9)
            x = rng.randint(1, 9)
            c = a * x + b
            thought = f"Solve for x. Subtract {b} from both sides: x = {c} - {b} = {x}."
        else:
            # Every number below comes straight from the question: the coefficient
            # `a` is repeated in both steps so it cannot drift mid-solution.
            move = f"{c} + {abs(b)}" if b < 0 else f"{c} - {b}"
            thought = (
                f"Solve for x. "
                f"Move {b} to the other side: {a}x = {_calc(move, c - b)}. "
                f"Divide: x = {_calc(f'{c - b} / {a}', x)}."
            )
        lhs = f"{a}x {'+' if b >= 0 else '-'} {abs(b)}"
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


def _smoke_test(seed: int = 0) -> None:
    """Print a small mixed sample for manual inspection of the <calc> format."""
    for gen in (generate_arithmetic, generate_fractions, generate_linear_equations):
        for ex in gen(seed=seed, n=7)[:7]:
            print(f"--- {ex.id} (difficulty {ex.difficulty})")
            print(ex.text + f"Answer: {ex.answer}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="generate ~20 examples and print them for manual inspection",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test(args.seed)
    else:
        parser.print_help()
