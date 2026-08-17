#!/usr/bin/env python
"""Best-of-N inference: sample the same prompt N times at temperature > 0,
verify each answer with real math, and keep the first correct one.

Jaiyu is 26M params and hallucinates digits under greedy decoding on some
arithmetic. Sampling gives it several tries; an external verifier (not the
model) checks which try is actually right.
"""

import argparse
import random
import re
import sys
from collections import Counter
from fractions import Fraction

import sympy
import torch
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from tokenizers import Tokenizer

from jaiyu.calculator import join_digits, pending_expr, result_span, space_digits
from jaiyu.model.config import load_config
from jaiyu.model.transformer import GPT

EOS_TOKEN = "<eos>"
MAX_TOOL_CALLS = 8  # a stuck model can emit <calc> forever; cap the loop
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def parse_args():
    p = argparse.ArgumentParser(description="Best-of-N verified sampling for Jaiyu.")
    p.add_argument("--checkpoint", default="checkpoints/v0.2_sft/step_500.pt")
    p.add_argument("--prompt", required=True, help="Question text, e.g. 'What is 25 + 60?'")
    p.add_argument("--raw", action="store_true",
                    help="Use the prompt verbatim instead of wrapping it in the training format.")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=None, help="Base seed; default picks a fresh one per run.")
    p.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# --- generation -------------------------------------------------------------

def load_model(checkpoint, device):
    config = load_config()
    model = GPT(config)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


# ponytail: recomputes the full prefix every step (no KV cache) -> O(n^2) in
# generated length. Fine at 26M params / 128 tokens on CPU; add a cache if
# num-samples or max-new-tokens grows enough to make this the bottleneck.
@torch.no_grad()
def generate(model, config, tokenizer, prompt, device, seed, temperature, max_new_tokens):
    gen = torch.Generator(device=device).manual_seed(seed)
    # The corpus spaces out every digit; an unspaced prompt is off-distribution.
    eos_id = tokenizer.get_vocab().get(EOS_TOKEN, 1)
    ids = tokenizer.encode(space_digits(prompt)).ids
    prompt_len = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    tool_calls = 0

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        if temperature == 0.0:
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=gen)
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eos_id:
            break
        # Training examples are "Thought: ...\nAnswer: X\n" -- two newlines.
        # Stop once both are out, or if the model starts a new example.
        completion = tokenizer.decode(idx[0, prompt_len:].tolist())

        # The model asked the calculator a question: answer it and let the model
        # read the result instead of guessing digits.
        expr = pending_expr(completion) if tool_calls < MAX_TOOL_CALLS else None
        if expr is not None:
            tool_calls += 1
            fed = tokenizer.encode(" " + result_span(expr)).ids
            idx = torch.cat(
                [idx, torch.tensor([fed], dtype=torch.long, device=device)], dim=1
            )
            continue

        if completion.count("\n") >= 2 or "Question:" in completion:
            break

    completion = tokenizer.decode(idx[0, prompt_len:].tolist())
    full_text = tokenizer.decode(idx[0].tolist())
    return full_text, completion


# --- answer extraction --------------------------------------------------

_NUM = r"-?\d+\.\d+|-?\d+/\d+|-?\d+"


def extract_model_answer(completion: str) -> str | None:
    completion = join_digits(completion)  # "6 0" -> "60"
    # Only ever look at the model's own completion, never the prompt/question
    # text -- prompts contain "=" and digits too (e.g. "Solve 4x - 8 = 12.").
    if "Answer:" in completion:
        tail = completion.split("Answer:")[-1]
    elif "=" in completion:
        tail = completion.rsplit("=", 1)[-1]
    else:
        tail = completion
    match = re.search(_NUM, tail)
    return match.group(0) if match else None


# --- verification ---------------------------------------------------------

def extract_prompt_expr(prompt: str) -> str | None:
    match = re.search(r"What is (.+)\?", prompt)
    if match:
        return match.group(1)
    match = re.search(r"Solve (.+)\.", prompt)
    if match:
        return match.group(1)
    return None


def _to_fraction(value: sympy.Expr) -> Fraction | None:
    value = sympy.nsimplify(value)
    if not value.is_Rational:
        return None
    return Fraction(int(value.p), int(value.q))


def verify_answer(prompt: str, answer: str | None) -> bool:
    expr = extract_prompt_expr(prompt)
    if expr is None or answer is None:
        return False
    try:
        given = Fraction(answer)
    except (ValueError, ZeroDivisionError):
        return False
    try:
        if "Solve" in prompt:
            x = sympy.symbols("x")
            lhs, rhs = expr.split("=")
            lhs_expr = parse_expr(lhs, transformations=_TRANSFORMATIONS)
            rhs_expr = parse_expr(rhs, transformations=_TRANSFORMATIONS)
            solutions = sympy.solve(sympy.Eq(lhs_expr, rhs_expr), x)
            return any(_to_fraction(sol) == given for sol in solutions)
        result = parse_expr(expr, transformations=_TRANSFORMATIONS)
        correct = _to_fraction(result)
        return correct is not None and correct == given
    except Exception as e:
        print(f"warning: could not verify {expr!r}: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model(args.checkpoint, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    prompt = args.prompt if args.raw else f"Question: {args.prompt}\nThought:"
    seed_base = args.seed if args.seed is not None else random.randrange(2**31)

    attempts = []
    for i in range(args.num_samples):
        full_text, completion = generate(model, config, tokenizer, prompt, device,
                                          seed=seed_base + i, temperature=args.temperature,
                                          max_new_tokens=args.max_new_tokens)
        answer = extract_model_answer(completion)
        correct = verify_answer(args.prompt, answer)
        attempts.append((full_text, answer, correct))

        if args.verbose:
            print(f"[attempt {i}] Answer: {answer} -> {'CORRECT' if correct else 'INCORRECT'}")

        if correct:
            print(join_digits(full_text))
            return

    print("WARNING: no sample verified correct; falling back to majority vote (UNVERIFIED).")
    votes = Counter(a for _, a, _ in attempts if a is not None)
    if not votes:
        print("No answer could be extracted from any attempt.")
        return
    winner = votes.most_common(1)[0][0]
    winning_response = next(r for r, a, _ in attempts if a == winner)
    print(join_digits(winning_response))


if __name__ == "__main__":
    main()
