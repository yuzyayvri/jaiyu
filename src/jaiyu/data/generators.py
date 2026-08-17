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


# Column names, least significant first. Four places cover every number the
# generators produce.
PLACES = ["ones", "tens", "hundreds", "thousands"]

# Digit counts to draw from, weighted toward the widths the model is weakest
# at while keeping the easy cases present so they are not forgotten.
WIDTHS = [1, 2, 2, 3, 3, 3, 4]


def _with_digits(rng: random.Random, width: int) -> int:
    """A random number with exactly `width` digits."""
    return rng.randint(10 ** (width - 1), 10 ** width - 1) if width > 1 else rng.randint(1, 9)


def _digits(n: int) -> list[int]:
    """Digits of `n`, least significant first."""
    return [int(d) for d in reversed(str(n))]


def _column_add(a: int, b: int) -> str:
    """Column-by-column addition trace, right to left, carrying as it goes.

    Written for any width rather than for two digits specifically: a model
    shown only the tens-and-ones version learns that shape instead of the
    procedure, and stops dead at three digits.
    """
    da, db = _digits(a), _digits(b)
    width = max(len(da), len(db))
    da += [0] * (width - len(da))
    db += [0] * (width - len(db))

    steps, columns, carry = [], [], 0
    for place in range(width):
        carry_in = carry
        raw = da[place] + db[place] + carry_in
        digit, carry = raw % 10, raw // 10
        expr = f"{da[place]} + {db[place]}"
        if carry_in:
            expr += f" + {carry_in}"
        steps.append(f"Add the {PLACES[place]}: {_calc(expr, raw)}.")
        if carry:
            steps.append(f"Write down {digit}, carry {carry}.")
        columns.append(digit * 10 ** place)
    if carry:
        columns.append(carry * 10 ** width)
        steps.append(f"The final carry gives a leading {carry}.")

    parts = [str(c) for c in reversed(columns) if c] or ["0"]
    steps.append(f"Combine: {_calc(' + '.join(parts), a + b)}.")
    return " ".join(steps)


def _column_sub(a: int, b: int) -> str:
    """Column-by-column subtraction trace with explicit borrowing (a >= b)."""
    da, db = _digits(a), _digits(b)
    da += [0] * (len(db) - len(da))
    db += [0] * (len(da) - len(db))

    steps, columns = [], []
    for place in range(len(da)):
        top = da[place]
        if top < db[place]:
            # Borrow from the next column that has something to give.
            source = next(p for p in range(place + 1, len(da)) if da[p] > 0)
            for p in range(place + 1, source):
                da[p] = 9  # each skipped 0 becomes 9
            da[source] -= 1
            top += 10
            steps.append(
                f"Subtract the {PLACES[place]}: {da[place]} - {db[place]} is not "
                f"possible, so borrow from the {PLACES[source]}, making it {top}."
            )
            steps.append(f"{_calc(f'{top} - {db[place]}', top - db[place])}.")
        else:
            steps.append(
                f"Subtract the {PLACES[place]}: "
                f"{_calc(f'{top} - {db[place]}', top - db[place])}."
            )
        columns.append((top - db[place]) * 10 ** place)

    parts = [str(c) for c in reversed(columns) if c] or ["0"]
    steps.append(f"Combine: {_calc(' + '.join(parts), a - b)}.")
    return " ".join(steps)


def _long_multiply(a: int, b: int) -> str:
    """Multiplication by splitting the larger factor into its place values."""
    if a < b:
        a, b = b, a
    parts = [d * 10 ** p for p, d in enumerate(_digits(a)) if d]
    if len(parts) == 1:
        return f"Multiply: {_calc(f'{a} * {b}', a * b)}."

    steps = [f"Split {a} into {' + '.join(str(p) for p in reversed(parts))}."]
    products = []
    for part in reversed(parts):
        products.append(part * b)
        steps.append(f"{_calc(f'{part} * {b}', part * b)}.")
    steps.append(
        f"Add the partial products: "
        f"{_calc(' + '.join(str(p) for p in products), a * b)}."
    )
    return " ".join(steps)


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
            # Widths are sampled rather than fixed at two digits: pre-training
            # left the model fluent to two digits and helpless at three, which
            # is what a corpus of only two-digit examples teaches.
            wa, wb = rng.choice(WIDTHS), rng.choice(WIDTHS)
            a, b = _with_digits(rng, wa), _with_digits(rng, wb)
            result = a + b
            thought = _column_add(a, b)
            difficulty = max(wa, wb)
        elif op == "-":
            wa = rng.choice(WIDTHS)
            a = _with_digits(rng, wa)
            b = rng.randint(1, a)  # b <= a keeps the difference non-negative
            result = a - b
            thought = _column_sub(a, b)
            difficulty = wa
        elif op == "*":
            wa = rng.choice([1, 1, 2, 2, 3])
            a = _with_digits(rng, wa)
            b = rng.randint(2, 12)
            result = a * b
            thought = _long_multiply(a, b)
            difficulty = 2 + (wa > 1) + (wa > 2)
        else:
            b = rng.randint(2, 12)
            a = b * _with_digits(rng, rng.choice([1, 1, 2, 2, 3]))
            result = a // b
            thought = f"Divide: {_calc(f'{a} / {b}', result)}."
            difficulty = 2 if a < 100 else 3

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
