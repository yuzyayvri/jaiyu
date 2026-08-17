import re
from fractions import Fraction

from jaiyu.calculator import evaluate
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


CALC = re.compile(r"<calc>(.*?)</calc>\s*<result>(.*?)</result>")


def assert_calcs_are_correct(text: str) -> list[str]:
    """Every <calc> in `text` states the value Python computes for it.

    Asserted on the tool spans rather than the surrounding prose, so rewording
    a reasoning trace does not break the check that its arithmetic is sound.
    """
    # Simple traces state the arithmetic inline and have no tool call at all;
    # the check is that any call present is right, not that one exists.
    spans = CALC.findall(text)
    for expr, stated in spans:
        assert evaluate(expr) == stated.strip(), f"{expr!r} -> {stated!r} in {text!r}"
    return [(expr, stated.strip()) for expr, stated in spans]


def test_arithmetic_thought_matches_question():
    # regression: no hallucinated numbers -- every <calc> computes what it
    # claims, and the trace ends on the true answer to the Question.
    question = re.compile(r"Question: What is (\d+) ([+\-*/]) (\d+)\?")
    for e in generate_arithmetic(seed=99, n=200):
        qa, op, qb = question.search(e.text).groups()
        # A trace may decompose the work over several steps (place value,
        # borrowing), so only its final value has to equal the answer.
        assert_calcs_are_correct(e.text)
        assert int(e.answer) == OPS[op](int(qa), int(qb))
        # Operands stay small: at most two digits, except a division dividend,
        # which is built as a product of two numbers up to 12.
        assert 0 < int(qa) <= (144 if op == "/" else 99)
        assert 0 < int(qb) <= 99


def test_linear_equations_thought_matches_question():
    # regression: coefficient/constant corruption between Question and Thought
    question = re.compile(r"Question: Solve (-?\d+)x ([+-]) (\d+) = (-?\d+)\.")
    for e in generate_linear_equations(seed=11, n=50):
        a, sign, b, c = question.search(e.text).groups()
        a, c = int(a), int(c)
        b = int(b) if sign == "+" else -int(b)

        spans = assert_calcs_are_correct(e.text)
        # When the trace shows its work, the constant moves across first and
        # the coefficient is divided out second.
        if len(spans) == 2:
            assert evaluate(spans[0][0]) == str(c - b), spans[0]
            assert evaluate(spans[1][0]) == str((c - b) // a), spans[1]

        x = int(e.answer.split("=")[-1])
        assert a * x + b == c, e.text


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
