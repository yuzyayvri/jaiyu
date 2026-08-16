"""Re-check the generated tool-use data by evaluating every printed expression."""

import random
import re
from fractions import Fraction

from mix_pretrain_data import gen_calc, gen_code, gen_formula, split_digits

CALC = re.compile(r"<calc>(.+?)</calc>\s*<result>(.+?)</result>")
CODE = re.compile(r"<output>(.+?)</output>")


DIV = re.compile(r"(\d+)\s*/\s*(\d+)")


def exact(expr: str) -> Fraction:
    """Evaluate with exact rational division instead of float division."""
    return Fraction(eval(DIV.sub(r"Fraction(\1,\2)", expr), {"Fraction": Fraction}))  # noqa: S307


def test_calc():
    rng = random.Random(26)
    for _ in range(2000):
        text = gen_calc(rng)
        pairs = CALC.findall(text)
        assert pairs, text
        for expr, result in pairs:
            assert Fraction(result.strip()) == exact(expr), (text, expr, result)


def test_code_and_formula():
    rng = random.Random(26)
    for _ in range(2000):
        text = gen_code(rng)
        out = CODE.search(text).group(1).strip()
        if text.startswith("sqrt"):
            n = int(re.search(r"sqrt\((\d+)\)", text).group(1))
            assert float(out) == __import__("math").sqrt(n), text
        elif text.startswith("fib"):
            n = int(re.search(r"fib\((\d+)\)", text).group(1))
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            assert int(out) == a, text
        else:
            expr = re.search(r"print\((.+?)\)", text).group(1)
            assert int(out) == eval(expr), text  # noqa: S307

        text = gen_formula(rng)
        expr, result = CALC.findall(text)[0]
        assert float(result) == eval(expr), text  # noqa: S307


def test_split_digits_idempotent():
    s = "34 + 15 = 49"
    assert split_digits(s) == split_digits(split_digits(s))


if __name__ == "__main__":
    test_calc()
    test_code_and_formula()
    test_split_digits_idempotent()
    print("all generator checks passed")
