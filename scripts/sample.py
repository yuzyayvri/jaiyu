#!/usr/bin/env python
"""Generate text from a trained Jaiyu checkpoint."""

import argparse

import torch
from tokenizers import Tokenizer

from jaiyu.calculator import join_digits, pending_expr, result_span, space_digits
from jaiyu.model.config import load_config
from jaiyu.model.transformer import GPT

EOS_TOKEN = "<eos>"
MAX_TOOL_CALLS = 8  # a stuck model can emit <calc> forever; cap the loop


def parse_args():
    p = argparse.ArgumentParser(description="Sample from a Jaiyu checkpoint.")
    p.add_argument("--checkpoint", default="outputs/checkpoints/digit-tokenizer/step_3000.pt",
                   help="Path to model checkpoint.")
    p.add_argument("--prompt", required=True, help="Prompt text.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    # A math problem has exactly one right answer, so the default decode is
    # greedy. Temperature and repetition penalty both make it worse: sampling
    # picks a wrong digit, and the penalty suppresses digits already in the
    # question (which are often the correct ones, e.g. "33 - 0 = 33").
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy.")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty.")
    p.add_argument("--raw", action="store_true",
                   help="Use the prompt verbatim instead of wrapping it in the training format.")
    p.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(26)

    temperature = args.temperature
    repetition_penalty = args.repetition_penalty

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer)

    config = load_config()
    # The tokenizer, not the yaml, is the authority on vocab size, exactly as
    # in training: a mismatch here is a shape error loading the embedding.
    config.vocab_size = tokenizer.get_vocab_size()

    model = GPT(config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Training checkpoints bundle the optimizer and step alongside the weights;
    # older ones held a bare state dict.
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device)
    model.eval()

    # Every training sequence looks like "Question: ...\nThought: ...". A bare
    # "34 + 15" is off-distribution and the model wanders into another topic.
    prompt = args.prompt if args.raw else f"Question: {args.prompt}\nThought:"
    # The corpus spaces out every digit; an unspaced prompt is off-distribution.
    ids = tokenizer.encode(space_digits(prompt)).ids
    prompt_len = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.get_vocab().get(EOS_TOKEN, 1)
    tool_calls = 0

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = model(idx_cond)

            # Apply repetition penalty to previously seen tokens (CTRL-style,
            # sign-aware: dividing a negative logit would make it larger)
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, -1, token_id] > 0:
                        logits[0, -1, token_id] /= repetition_penalty
                    else:
                        logits[0, -1, token_id] *= repetition_penalty

            if temperature == 0.0:
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_id:
                break

            # The model asked the calculator a question: answer it and let the
            # model read the result instead of guessing digits.
            completion = tokenizer.decode(idx[0, prompt_len:].tolist())
            expr = pending_expr(completion) if tool_calls < MAX_TOOL_CALLS else None
            if expr is not None:
                tool_calls += 1
                fed = tokenizer.encode(" " + result_span(expr)).ids
                idx = torch.cat(
                    [idx, torch.tensor([fed], dtype=torch.long, device=device)], dim=1
                )

    output_ids = idx[0].tolist()
    print(join_digits(tokenizer.decode(output_ids)))


if __name__ == "__main__":
    main()
