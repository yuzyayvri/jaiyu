"""The calculator tool Jaiyu calls with `<calc> expr </calc>`.

The model emits an expression and the runtime -- not the model -- computes the
value and feeds back `<result> value </result>`. Everything is exact rational
arithmetic (`fractions.Fraction`), so `1/3 + 1/6` is `1/2`, never `0.4999...`.

Model output is untrusted input, so expressions are whitelisted down to digits
and the four operators before evaluation.
"""

from __future__ import annotations

import re
from fractions import Fraction

CALC_OPEN = "<calc>"
CALC_CLOSE = "</calc>"

# Only these characters may appear in an expression. No names, no attributes,
# no exponent operator (which would let `2**999999` hang the runtime).
_ALLOWED = re.compile(r"^[0-9+\-*/(). ]+$")
_NUMBER = re.compile(r"\d+\.\d+|\.\d+|\d+")
_CALC_SPAN = re.compile(r"<calc>(.*?)</calc>(\s*<result>)?", re.DOTALL)

# The pre-training corpus spaces out every digit ("4 5 + 1 5"), so expressions
# may arrive in that form; digits separated only by whitespace are one number.
_SPACED_DIGITS = re.compile(r"(?<=\d)\s+(?=\d)")

MAX_EXPR_LEN = 200


def join_digits(text: str) -> str:
    """Undo the corpus digit spacing: "6 0" -> "60". Leaves other spacing alone."""
    return _SPACED_DIGITS.sub("", text)


def space_digits(text: str) -> str:
    """Digit spacing used by the corpus: "60" -> "6 0"."""
    return re.sub(r"(?<=\d)(?=\d)", " ", text)


def evaluate(expr: str) -> str | None:
    """Exact value of `expr` as a string, or None if it is unsafe or invalid.

    >>> evaluate("1/3 + 1/6")
    '1/2'
    """
    expr = expr.strip()
    if not expr or len(expr) > MAX_EXPR_LEN:
        return None
    if not _ALLOWED.match(expr):
        return None
    expr = join_digits(expr)
    if "**" in expr or "//" in expr:
        return None
    # Every literal becomes a Fraction, so `/` stays exact and `0.1` is 1/10.
    py = _NUMBER.sub(lambda m: f"Fraction('{m.group(0)}')", expr)
    try:
        value = eval(py, {"__builtins__": {}}, {"Fraction": Fraction})  # noqa: S307
    except Exception:
        return None
    if not isinstance(value, Fraction):
        return None
    return str(value)


def result_span(expr: str) -> str:
    """The `<result> ... </result>` span to feed back for `expr`.

    The result is written in the same digit style as the expression, so a model
    trained on spaced digits reads back spaced digits.
    """
    value = evaluate(expr)
    value = value if value is not None else "error"
    if _SPACED_DIGITS.search(expr):
        value = space_digits(value)
    return f"<result> {value} </result>"


def fill_results(text: str) -> str:
    """Compute every `<calc>` in `text` that has no `<result>` yet.

    Used offline (eval, data repair); generation feeds results back one span at
    a time so the model actually reads them.
    """

    def replace(match: re.Match[str]) -> str:
        if match.group(2):  # already followed by a <result>
            return match.group(0)
        return f"{CALC_OPEN}{match.group(1)}{CALC_CLOSE} {result_span(match.group(1))}"

    return _CALC_SPAN.sub(replace, text)


def pending_expr(text: str) -> str | None:
    """The expression of a just-closed `<calc>` at the end of `text`, if any."""
    if not text.rstrip().endswith(CALC_CLOSE):
        return None
    head = text.rstrip()[: -len(CALC_CLOSE)]
    if CALC_OPEN not in head:
        return None
    return head.rsplit(CALC_OPEN, 1)[-1]


def _self_check() -> None:
    assert evaluate("4 + 5") == "9"
    assert evaluate("1/3 + 1/6") == "1/2"
    assert evaluate("30/35 + 63/35") == "93/35"
    assert evaluate("3.14159 * 2 * 2") == str(Fraction("3.14159") * 4)
    assert evaluate("-9 / 3") == "-3"
    assert evaluate("(2 + 3) * 4") == "20"
    # invalid / unsafe input returns None rather than raising or executing
    assert evaluate("1/0") is None
    assert evaluate("2**900") is None
    assert evaluate("__import__('os').system('ls')") is None
    assert evaluate("x + 1") is None
    assert evaluate("") is None
    assert evaluate("9" * (MAX_EXPR_LEN + 1)) is None
    assert evaluate("1 +") is None

    # spaced-digit form (pre-training style) round-trips in the same style
    assert evaluate("4 5 + 1 5") == "60"
    assert result_span(" 4 5 + 1 5 ") == "<result> 6 0 </result>"
    assert result_span("4 + 5") == "<result> 9 </result>"
    assert result_span("1/0") == "<result> error </result>"

    text = "Add: <calc> 2 + 3 </calc>. Then <calc> 10 / 4 </calc> <result> 5/2 </result>."
    filled = fill_results(text)
    assert "<calc> 2 + 3 </calc> <result> 5 </result>" in filled
    assert filled.count("<result>") == 2, filled  # the existing one is untouched

    assert pending_expr("Add: <calc> 4 + 5 </calc>") == " 4 + 5 "
    assert pending_expr("Add: <calc> 4 + 5 </calc> <result> 9 </result>") is None
    assert pending_expr("Add: <calc> 4 + 5") is None
    print("ok")


if __name__ == "__main__":
    _self_check()
