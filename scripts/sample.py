#!/usr/bin/env python
"""Generate text from a trained Jaiyu checkpoint."""

import argparse

import torch
from tokenizers import Tokenizer

from jaiyu.model.config import GPTConfig
from jaiyu.model.transformer import GPT

import random

EOS_ID = 50256


def parse_args():
    p = argparse.ArgumentParser(description="Sample from a Jaiyu checkpoint.")
    p.add_argument("--checkpoint", default="outputs/checkpoints/step_1000.pt", help="Path to model checkpoint.")
    p.add_argument("--prompt", required=True, help="Prompt text.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    p.add_argument("--repetition-penalty", type=float, default=1.3, help="Repetition penalty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(26)

    temperature = args.temperature
    repetition_penalty = args.repetition_penalty

    device = torch.device("cuda")

    config = GPTConfig()
    model = GPT(config)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_pretrained("gpt2")

    ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    temperature = 0.8
    repetition_penalty = 1.3

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = model(idx_cond)

            # Apply repetition penalty to previously seen tokens
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, -1, token_id] /= repetition_penalty

            # Apply temperature
            logits = logits / temperature

            probs = torch.softmax(logits[:, -1, :], dim=-1)

            # Sample from distribution instead of greedy argmax
            next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == EOS_ID:
                break

    output_ids = idx[0].tolist()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
