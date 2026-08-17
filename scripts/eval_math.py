#!/usr/bin/env python
"""Exact-answer accuracy per category for a Jaiyu checkpoint.

Loss says the model fits the distribution; this says whether it gets the
arithmetic right, which is the signal the project actually cares about. Every
expected answer is computed in Python, so the harness cannot be wrong about
what the answer is.

Problems are generated from a seed rather than read from a file, so an eval
run never touches training data on disk and the operand ranges can be pushed
past what training covered (3-digit and up) to separate memorised facts from
learned procedure.
"""

import argparse
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jaiyu.calculator import join_digits, space_digits  # noqa: E402
from jaiyu.model.config import load_config  # noqa: E402
from jaiyu.model.transformer import GPT  # noqa: E402

# name -> (operand low, operand high, operator)
CATEGORIES = {
    "add_1digit": (0, 9, "+"),
    "add_2digit": (10, 99, "+"),
    "add_3digit": (100, 999, "+"),
    "sub_1digit": (0, 9, "-"),
    "sub_2digit": (10, 99, "-"),
    "sub_3digit": (100, 999, "-"),
    "mul_small": (2, 12, "*"),
    "mul_2digit": (10, 99, "*"),
    "div_exact": (2, 12, "/"),
}

_RESULT = re.compile(r"<result>(.*?)(?:</result>|$)", re.DOTALL)
_NUMBER = re.compile(r"-?\d+")


def truth(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a // b


def make_problem(rng: random.Random, lo: int, hi: int, op: str) -> tuple[int, int]:
    """Operands for one problem, arranged so the answer stays an integer."""
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    if op == "-" and b > a:
        a, b = b, a
    if op == "/":  # build the dividend so the division comes out exact
        a = a * b
    return a, b


def numbers_in(text: str) -> list[int]:
    """Every number in corpus-style spaced-digit text.

    "8 4 - 2 6   5 8" is three numbers, not one: single spaces separate the
    digits of one number, wider gaps separate numbers. Collapsing all
    whitespace would read that as 8426558.
    """
    text = re.sub(r"\s{2,}", "\x00", text)          # wide gap: a real boundary
    text = join_digits(text)                         # single spaces: within a number
    # A minus between two numbers is the subtraction operator being echoed back,
    # not a sign: "84 - 26" is two positive numbers, while a leading "- 7" is
    # negative seven.
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "\x00", text)
    text = re.sub(r"-\s+(?=\d)", "-", text)
    return [int(n) for n in _NUMBER.findall(text)]


def extract_answer(out: str, a: int, b: int) -> int | None:
    """The model's answer, or None if it never produced one.

    Two output shapes appear in the pre-training corpus and both are accepted:
    a bare "= 58", and the tool form "<calc> 84 - 26 </calc> <result> 58
    </result>". The tool form re-states the operands before answering, so a
    naive "first number" read returns an operand; leading numbers that merely
    echo the question are skipped.
    """
    # Generation runs for a fixed number of tokens, so the model usually
    # finishes the answer and starts inventing a new problem. Everything from
    # the first newline or <eos> onward belongs to that next problem, including
    # any <result> in it, and must not be read as this answer.
    out = re.split(r"\n|<eos>", out, maxsplit=1)[0]

    span = _RESULT.search(out)
    text = span.group(1) if span else out

    numbers = numbers_in(text)
    # The tool form restates the whole question ("84 - 26") before answering.
    # Only that complete restatement is skipped: dropping any leading number
    # that merely matches an operand would discard the answer to "8 + 0".
    if not span and numbers[:2] == [a, b]:
        numbers = numbers[2:]
    return numbers[0] if numbers else None


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
             device, block_size: int) -> list[str]:
    """Greedy continuation of each prompt.

    Prompts are batched only with others of the same token length. The model
    takes no attention mask, so padding a short prompt would put pad tokens in
    the context and change what it predicts; equal-length batches avoid the
    question entirely.
    """
    encoded = [tokenizer.encode(space_digits(p)).ids for p in prompts]

    buckets: dict[int, list[int]] = {}
    for i, ids in enumerate(encoded):
        buckets.setdefault(len(ids), []).append(i)

    out: list[str | None] = [None] * len(prompts)
    for width, indices in buckets.items():
        x = torch.tensor([encoded[i] for i in indices], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits, _ = model(x[:, -block_size:])
            x = torch.cat([x, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        for i, row in zip(indices, x):
            out[i] = tokenizer.decode(row[width:].tolist(), skip_special_tokens=False)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    p.add_argument("--num-problems", type=int, default=100,
                   help="Problems per category.")
    p.add_argument("--max-new-tokens", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                   help=f"Subset of: {' '.join(CATEGORIES)}")
    p.add_argument("--show-failures", type=int, default=3,
                   help="Print this many wrong answers per category.")
    args = p.parse_args()

    for name in args.categories:
        if name not in CATEGORIES:
            raise SystemExit(f"unknown category {name!r}; choose from {list(CATEGORIES)}")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = load_config()
    config.vocab_size = tokenizer.get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"checkpoint {args.checkpoint} (step {step}), "
          f"{args.num_problems} problems per category\n")
    print(f"{'category':<14}{'accuracy':>14}   failures")

    totals = [0, 0]
    for name in args.categories:
        lo, hi, op = CATEGORIES[name]
        rng = random.Random(args.seed)
        problems = [make_problem(rng, lo, hi, op) for _ in range(args.num_problems)]

        correct, failures = 0, []
        for start in range(0, len(problems), args.batch_size):
            batch = problems[start:start + args.batch_size]
            outs = generate(model, tokenizer, [f"{a} {op} {b} = " for a, b in batch],
                            args.max_new_tokens, device, config.block_size)
            for (a, b), out in zip(batch, outs):
                got, want = extract_answer(out, a, b), truth(a, b, op)
                if got == want:
                    correct += 1
                elif len(failures) < args.show_failures:
                    failures.append(f"{a}{op}{b}={got} (want {want})")

        totals[0] += correct
        totals[1] += len(problems)
        pct = 100 * correct / len(problems)
        print(f"{name:<14}{correct:>5}/{len(problems)} {pct:5.1f}%   {'; '.join(failures)}")

    print(f"\n{'OVERALL':<14}{totals[0]:>5}/{totals[1]} {100 * totals[0] / totals[1]:5.1f}%")


if __name__ == "__main__":
    main()
