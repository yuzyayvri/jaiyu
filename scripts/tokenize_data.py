"""Tokenize and pack Jaiyu synthetic math data into fixed-length chunks."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tokenizers import Tokenizer

# Same packing constants and digit spacing as the pre-training corpus, so SFT
# text is formatted exactly like what the model was pre-trained on.
from mix_pretrain_data import EOS_TOKEN, PAD_TOKEN, SEQ_LEN, split_digits

IN_DIR = Path("data/intermediate/synthetic")
OUT_DIR = Path("data/processed/tokenized")


def load_examples(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_and_pack(examples: list[dict], tokenizer: Tokenizer) -> np.ndarray:
    vocab = tokenizer.get_vocab()
    eos_id, pad_id = vocab[EOS_TOKEN], vocab[PAD_TOKEN]

    ids: list[int] = []
    for ex in examples:
        full_text = ex["text"] + "Answer: " + ex["answer"] + "\n"
        ids.extend(tokenizer.encode(split_digits(full_text)).ids)
        ids.append(eos_id)

    n_chunks = (len(ids) + SEQ_LEN - 1) // SEQ_LEN
    padded_len = n_chunks * SEQ_LEN
    ids.extend([pad_id] * (padded_len - len(ids)))

    arr = np.array(ids, dtype=np.uint16).reshape(n_chunks, SEQ_LEN)
    return arr


def process(split: str, tokenizer: Tokenizer) -> None:
    examples = load_examples(IN_DIR / f"{split}.jsonl")
    arr = tokenize_and_pack(examples, tokenizer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.npy"
    np.save(out_path, arr)

    print(f"{split}: {arr.size} tokens, shape {arr.shape} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    process("train", tokenizer)
    process("eval", tokenizer)


if __name__ == "__main__":
    main()
