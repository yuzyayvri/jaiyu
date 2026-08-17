"""Checks for the parts of the math eval that can be wrong silently.

A broken answer extractor reports a working model as broken (and the reverse),
so it is worth more scrutiny than the generation loop around it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_math import CATEGORIES, extract_answer, make_problem, truth  # noqa: E402


def test_extracts_from_result_span():
    out = " <calc> 8 4 - 2 6 </calc> <result> 5 8 </result>"
    assert extract_answer(out, 84, 26) == 58


def test_extracts_from_bare_continuation():
    assert extract_answer("5 8 \n more text", 84, 26) == 58


def test_skips_operands_echoed_before_the_answer():
    # The corpus restates the question before answering; the operands are not
    # the answer, which is the bug that made a working model look broken.
    assert extract_answer(" 8 4 - 2 6   5 8", 84, 26) == 58


def test_keeps_answer_equal_to_an_operand():
    # 5 - 0 = 5: the answer coincides with an operand and must survive.
    assert extract_answer(" <result> 5 </result>", 5, 0) == 5


def test_reads_only_the_first_line_without_a_result_span():
    assert extract_answer("4 2 \n 9 9 9", 17, 25) == 42


def test_negative_answers():
    assert extract_answer(" <result> - 7 </result>", 3, 10) == -7


def test_missing_answer_is_none():
    assert extract_answer(" \n", 1, 2) is None


def test_truth_matches_python():
    assert truth(84, 26, "-") == 58
    assert truth(7, 8, "*") == 56
    assert truth(12, 4, "/") == 3


def test_generated_problems_have_integer_answers():
    import random

    for name, (lo, hi, op) in CATEGORIES.items():
        rng = random.Random(26)
        for _ in range(200):
            a, b = make_problem(rng, lo, hi, op)
            assert lo <= b <= hi, name
            if op == "-":
                assert truth(a, b, op) >= 0, name          # no negatives by construction
            if op == "/":
                assert a % b == 0 and truth(a, b, op) == a / b, name


def test_seed_makes_problems_reproducible():
    import random

    lo, hi, op = CATEGORIES["add_2digit"]
    first = [make_problem(random.Random(26), lo, hi, op) for _ in range(5)]
    second = [make_problem(random.Random(26), lo, hi, op) for _ in range(5)]
    assert first == second


def test_ignores_a_result_from_the_next_invented_problem():
    # Generation does not stop at the answer; the model starts a new problem
    # whose <result> tag would otherwise be read as this answer.
    out = "6 \n<eos> 1 0 * 1 2 = <calc> 1 0 * 1 2 </calc> <result>"
    assert extract_answer(out, 3, 3) == 6


def test_stops_at_eos_without_a_newline():
    assert extract_answer("2 0 <eos> 9 9 + 9 9 = <result> 1 9 8 </result>", 15, 5) == 20


def test_bare_answer_equal_to_an_operand_survives():
    # "8 + 0 = 8": the answer repeats an operand and is not an echo.
    assert extract_answer("8 ", 8, 0) == 8
    assert extract_answer("2 0 ", 20, 0) == 20
