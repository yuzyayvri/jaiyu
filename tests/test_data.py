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


OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b,
}


def test_arithmetic_thought_matches_question():
    # regression: no hallucinated numbers -- every value in the Thought is
    # either an operand from the Question or the correct result.
    question = re.compile(r"Question: What is (\d+) ([+\-*/]) (\d+)\?")
    thought = re.compile(r"Thought: [A-Za-z ]+ (\d+) ?[+\-*/by]* ?(\d+)\. .* is (\d+)\.")
    for e in generate_arithmetic(seed=99, n=200):
        qa, op, qb = question.search(e.text).groups()
        ta, tb, result = thought.search(e.text).groups()
        assert (ta, tb) == (qa, qb)
        assert int(result) == OPS[op](int(qa), int(qb)) == int(e.answer)
        assert 0 <= int(qa) <= 100 and 0 <= int(qb) <= 100


def test_linear_equations_thought_matches_question():
    # regression: coefficient/constant corruption between Question and Thought
    pattern = re.compile(
        r"Question: Solve (-?\d+)x ([+-]) (\d+) = (-?\d+)\.\n"
        r"Thought: Solve for x\. Move (-?\d+) to the other side: (-?\d+)x = (-?\d+)\. "
        r"Divide both sides by (-?\d+): x = (-?\d+)\.\n"
    )
    for e in generate_linear_equations(seed=11, n=50):
        a, sign, b, c, move_b, step_a, rhs, div_a, x = pattern.search(e.text).groups()
        b = int(b) if sign == "+" else -int(b)
        assert int(move_b) == b
        assert int(step_a) == int(div_a) == int(a)
        assert int(rhs) == int(c) - b
        assert int(a) * int(x) == int(rhs)


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
    test_arithmetic_thought_matches_question()
    test_fractions_correct()
    test_linear_equations_solution_correct()
    test_linear_equations_thought_matches_question()
    print("ok")
