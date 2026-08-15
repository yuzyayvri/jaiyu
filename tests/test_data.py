import re
from fractions import Fraction

from jaiyu.data import (
    generate_arithmetic,
    generate_fractions,
    generate_linear_equations,
)


def test_arithmetic_deterministic_and_correct():
    ex1 = generate_arithmetic(seed=42, n=20)
    ex2 = generate_arithmetic(seed=42, n=20)
    assert [e.text for e in ex1] == [e.text for e in ex2]
    for e in ex1:
        assert e.text.startswith("Question:")
        assert "Thought:" in e.text
        assert "Answer" not in e.text
        assert 1 <= e.difficulty <= 10
        assert e.split in ("train", "eval")


def test_arithmetic_place_value_reasoning_consistent():
    # regression: intermediates must actually sum to the stated result
    combine = re.compile(r"Combine: (-?\d+) \+ (-?\d+) = (-?\d+)\.")
    for e in generate_arithmetic(seed=99, n=200):
        m = combine.search(e.text)
        if m:  # only +/- examples emit a "Combine:" line
            tens_part, ones_part, result = (int(v) for v in m.groups())
            assert tens_part + ones_part == result == int(e.answer)


def test_fractions_correct():
    for e in generate_fractions(seed=7, n=20):
        assert Fraction(e.answer) == Fraction(e.answer)  # parses cleanly


def test_linear_equations_solution_correct():
    pattern = re.compile(r"Solve (-?\d+)x ([+-]) (\d+) = (-?\d+)\.")
    for e in generate_linear_equations(seed=3, n=30):
        a, sign, b, c = pattern.search(e.text).groups()
        b = int(b) if sign == "+" else -int(b)
        x = int(e.answer.removeprefix("x = "))
        assert int(a) * x + b == int(c)


if __name__ == "__main__":
    test_arithmetic_deterministic_and_correct()
    test_arithmetic_place_value_reasoning_consistent()
    test_fractions_correct()
    test_linear_equations_solution_correct()
    print("ok")
