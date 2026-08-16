#!/usr/bin/env python
"""Train Jaiyu locally on a single 8GB VRAM GPU."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
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
    p.add_argument("--no-eval", action="store_true", help="Disable evaluation entirely.")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--keep-checkpoints", type=int, default=3,
                   help="Delete all but the N most recent checkpoints. 0 keeps every one.")
    p.add_argument("--output-dir", default="outputs/checkpoints")
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--resume-from", default=None, help="Checkpoint path to load weights (and optimizer state) from.")
    p.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json",
                   help="Tokenizer whose vocab_size overrides the model config.")
    p.add_argument("--fp16", action="store_true", help="fp16 mixed precision + GradScaler (for T4-class GPUs).")
    p.add_argument("--warmup-steps", type=int, default=1000)
    return p.parse_args()


def lr_at(step: int, args) -> float:
    """Linear warmup, then cosine decay to 10% of peak."""
    if step <= args.warmup_steps:
        return args.learning_rate * step / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(1.0, progress)
    return args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
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

    # The tokenizer, not the yaml, is the authority on vocab size: a mismatch
    # is an out-of-bounds embedding lookup that faults the GPU queue.
    if Path(args.tokenizer).exists():
        vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()
        config.vocab_size = vocab_size
        print(f"Tokenizer loaded: {args.tokenizer}, vocab_size: {vocab_size}", flush=True)
    else:
        print(f"Tokenizer {args.tokenizer} not found, using config vocab_size {config.vocab_size}", flush=True)

    train_ds = TokenDataset(args.train_data, block_size=config.block_size)
    assert len(train_ds) > 0, "Training dataset is empty"

    # Missing eval data is an error rather than a silent downgrade: a long run
    # that quietly trains blind is only discovered once it is over.
    do_eval = not args.no_eval
    if do_eval and not Path(args.eval_data).exists():
        raise SystemExit(
            f"{args.eval_data} not found. Build it with scripts/make_pretrain_data.py "
            f"and scripts/pack_jsonl.py, or pass --no-eval to train without it."
        )

    eval_loader = None
    datasets = [("train", train_ds)]
    if do_eval:
        eval_ds = TokenDataset(args.eval_data, block_size=config.block_size)
        assert len(eval_ds) > 0, "Eval dataset is empty"
        datasets.append(("eval", eval_ds))
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    else:
        print("--no-eval set, training without evaluation.", flush=True)

    # Out-of-range ids are an out-of-bounds embedding lookup: on CPU that's an
    # IndexError, on ROCm it faults the GPU queue with an unreadable HSA error.
    for name, ds in datasets:
        max_id = int(ds.data.max())
        assert max_id < config.vocab_size, (
            f"{name} data contains token id {max_id} but vocab_size is "
            f"{config.vocab_size}; re-tokenize or fix the config"
        )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    train_iter = cycle(train_loader)

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        if "model" in ckpt:  # new format: model + optimizer + step
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
        else:  # legacy checkpoints held a bare state_dict
            model.load_state_dict(ckpt)
        print(f"Resumed from {args.resume_from} at step {start_step}", flush=True)
    else:
        print("Starting from random initialization", flush=True)

    # bf16 has the same exponent range as fp32 and needs no scaler; fp16 does.
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(device.type, enabled=args.fp16 and device.type == "cuda")

    def save(step):
        ckpt_path = output_dir / f"step_{step}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, ckpt_path)
        print(f"saved checkpoint to {ckpt_path}", flush=True)
        # Each checkpoint carries optimizer state, so it is ~3x the model size.
        # A long run would otherwise fill the disk with checkpoints nobody
        # reads; only the most recent ones are ever used to resume.
        if args.keep_checkpoints > 0:
            saved = sorted(output_dir.glob("step_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
            for old in saved[:-args.keep_checkpoints]:
                old.unlink()

    model.train()
    window_start = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            print(f"step {step}: train loss {loss.item():.4f} lr {lr:.2e}", flush=True)

        if step % 100 == 0:
            elapsed = time.time() - window_start
            tps = args.batch_size * config.block_size * 100 / elapsed
            eta = (args.max_steps - step) * elapsed / 100
            print(
                f"step {step}: {tps:,.0f} tokens/sec, eta {eta / 60:.1f} min",
                flush=True,
            )
            window_start = time.time()

        if do_eval and step % args.eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device, amp_dtype)
            print(f"step {step}: eval loss {eval_loss:.4f}", flush=True)

        if step % args.save_interval == 0:
            save(step)


if __name__ == "__main__":
    main()
