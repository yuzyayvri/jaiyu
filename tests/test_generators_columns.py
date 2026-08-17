"""Every number in a reasoning trace must be one Python computed.

The traces are what the model imitates, so a wrong intermediate step teaches
wrong arithmetic even when the final answer is right. These checks read the
<calc> spans out of a trace and verify each one independently.
"""

import re

import pytest

from jaiyu.calculator import evaluate
from jaiyu.data.generators import _column_add, _column_sub, _long_multiply

CALC = re.compile(r"<calc>(.*?)</calc>\s*<result>(.*?)</result>")


def assert_trace_is_sound(trace: str, expected: int) -> None:
    spans = CALC.findall(trace)
    assert spans, f"no <calc> in {trace!r}"
    for expr, stated in spans:
        assert evaluate(expr) == stated.strip(), f"{expr!r} -> {stated!r} in {trace!r}"
    # The last span is the one that states the answer.
    assert int(spans[-1][1].strip()) == expected, trace


@pytest.mark.parametrize("a,b", [
    (7, 8), (17, 25), (84, 26), (99, 99), (100, 1), (137, 489), (865, 307),
    (999, 999), (1234, 8766), (5, 9995), (1000, 2000), (9999, 1),
])
def test_column_add(a, b):
    assert_trace_is_sound(_column_add(a, b), a + b)


@pytest.mark.parametrize("a,b", [
    (8, 3), (26, 17), (84, 26), (100, 1), (1000, 1), (865, 307), (775, 310),
    (999, 999), (1000, 999), (5000, 4999), (9999, 1111), (10, 9),
])
def test_column_sub(a, b):
    assert_trace_is_sound(_column_sub(a, b), a - b)


@pytest.mark.parametrize("a,b", [
    (7, 8), (12, 7), (97, 8), (86, 12), (123, 4), (999, 9), (100, 5), (205, 7),
])
def test_long_multiply(a, b):
    assert_trace_is_sound(_long_multiply(a, b), a * b)


def test_borrowing_across_zeros():
    # 1000 - 1 borrows through two zero columns, the case a two-digit-only
    # implementation never has to handle.
    trace = _column_sub(1000, 1)
    assert_trace_is_sound(trace, 999)
    assert "borrow" in trace


def test_carry_extends_the_width():
    # 999 + 1 grows a new leading digit.
    assert_trace_is_sound(_column_add(999, 1), 1000)


def test_equal_operands_subtract_to_zero():
    assert_trace_is_sound(_column_sub(500, 500), 0)


def test_random_traces_across_all_widths():
    # Exhaustive-ish sweep: the hand-picked cases above cannot cover every
    # borrow/carry pattern, and a wrong trace is invisible until a model has
    # already learned it.
    import random

    rng = random.Random(26)
    for _ in range(2000):
        a, b = rng.randint(1, 9999), rng.randint(1, 9999)
        assert_trace_is_sound(_column_add(a, b), a + b)
        assert_trace_is_sound(_column_sub(max(a, b), min(a, b)), abs(a - b))

        multiplier = rng.randint(2, 12)
        assert_trace_is_sound(_long_multiply(a, multiplier), a * multiplier)
