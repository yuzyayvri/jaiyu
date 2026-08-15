#!/usr/bin/env python
"""Train Jaiyu locally on a single 8GB VRAM GPU."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jaiyu.model.config import load_config
from jaiyu.model.transformer import GPT


class TokenDataset(Dataset):
    def __init__(self, path: str, block_size: int = 512):
        tokens = np.load(path)

        # If the tokenizer saved a 2D array (N, 512), flatten it to 1D
        if tokens.ndim == 2:
            tokens = tokens.flatten()

        # Truncate to the nearest multiple of (block_size + 1); leftover
        # tokens that don't fill a full sequence are discarded.
        n_seq = (len(tokens) // (block_size + 1)) * (block_size + 1)
        self.data = torch.from_numpy(tokens[:n_seq].astype(np.int64)).view(-1, block_size + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="data/processed/tokenized/train.npy")
    p.add_argument("--eval-data", default="data/processed/tokenized/eval.npy")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--output-dir", default="outputs/checkpoints")
    p.add_argument("--seed", type=int, default=26)
    return p.parse_args()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config()
    train_ds = TokenDataset(args.train_data, block_size=config.block_size)
    eval_ds = TokenDataset(args.eval_data, block_size=config.block_size)
    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(eval_ds) > 0, "Eval dataset is empty"
    # Out-of-range ids are an out-of-bounds embedding lookup: on CPU that's an
    # IndexError, on ROCm it faults the GPU queue with an unreadable HSA error.
    for name, ds in (("train", train_ds), ("eval", eval_ds)):
        max_id = int(ds.data.max())
        assert max_id < config.vocab_size, (
            f"{name} data contains token id {max_id} but vocab_size is "
            f"{config.vocab_size}; re-tokenize or fix the config"
        )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    train_iter = cycle(train_loader)

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # bf16 has the same exponent range as fp32, so no gradient scaler is needed.
    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step}: train loss {loss.item():.4f}")

        if step % args.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device)
            print(f"step {step}: eval loss {eval_loss:.4f}")

        if step % args.save_interval == 0:
            ckpt_path = output_dir / f"step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
