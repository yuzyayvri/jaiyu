"""Tokenize and pack Jaiyu fine-tuning data into fixed-length chunks.

Alongside the tokens, writes a boolean mask marking which positions the model
is trained to predict. Calculator results are excluded: at inference the
runtime computes them and feeds them back, so training the model to produce
them teaches it to guess digits rather than call the tool. Everything else --
the question, the reasoning, the `<calc>` expression, the final answer -- is
trained on.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tokenizers import Tokenizer

# Same packing constants and digit spacing as the pre-training corpus, so the
# fine-tuning text is formatted exactly like what the model was pre-trained on.
from mix_pretrain_data import EOS_TOKEN, PAD_TOKEN, SEQ_LEN, split_digits

IN_DIR = Path("data/intermediate/synthetic")
OUT_DIR = Path("data/processed/tokenized")

# The whole span, tags included: the model should stop after "</calc>" and let
# the runtime supply this.
RESULT_SPAN = re.compile(r"<result>.*?</result>", re.DOTALL)


def load_examples(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Generate it with scripts/make_synthetic_data.py first."
        )
    with path.open() as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if not examples:
        raise SystemExit(f"{path} is empty.")
    return examples


def encode_with_mask(text: str, tokenizer: Tokenizer) -> tuple[list[int], list[bool]]:
    """Token ids for `text`, plus which of them are trained on.

    The text is encoded in alternating segments so a result span keeps its own
    tokens: the tags are special tokens, so every segment boundary falls on a
    token boundary and piecewise encoding matches encoding the whole string.
    """
    ids: list[int] = []
    mask: list[bool] = []

    position = 0
    for span in RESULT_SPAN.finditer(text):
        for chunk, trained in ((text[position:span.start()], True), (span.group(0), False)):
            if chunk:
                chunk_ids = tokenizer.encode(split_digits(chunk)).ids
                ids.extend(chunk_ids)
                mask.extend([trained] * len(chunk_ids))
        position = span.end()

    tail = text[position:]
    if tail:
        tail_ids = tokenizer.encode(split_digits(tail)).ids
        ids.extend(tail_ids)
        mask.extend([True] * len(tail_ids))

    return ids, mask


def tokenize_and_pack(examples: list[dict], tokenizer: Tokenizer,
                      mask_results: bool) -> tuple[np.ndarray, np.ndarray]:
    vocab = tokenizer.get_vocab()
    eos_id, pad_id = vocab[EOS_TOKEN], vocab[PAD_TOKEN]

    ids: list[int] = []
    mask: list[bool] = []
    for ex in examples:
        # Replayed pre-training text has no answer and takes no wrapper; it is
        # here to be modelled exactly as it was during pre-training.
        full_text = ex["text"]
        if ex.get("answer"):
            full_text += "Answer: " + ex["answer"] + "\n"
        if mask_results:
            ex_ids, ex_mask = encode_with_mask(full_text, tokenizer)
        else:
            ex_ids = tokenizer.encode(split_digits(full_text)).ids
            ex_mask = [True] * len(ex_ids)
        ids.extend(ex_ids)
        mask.extend(ex_mask)
        ids.append(eos_id)
        mask.append(True)  # ending the example is a prediction worth learning

    n_chunks = (len(ids) + SEQ_LEN - 1) // SEQ_LEN
    padding = n_chunks * SEQ_LEN - len(ids)
    ids.extend([pad_id] * padding)
    mask.extend([False] * padding)  # never train on padding

    return (np.array(ids, dtype=np.uint16).reshape(n_chunks, SEQ_LEN),
            np.array(mask, dtype=bool).reshape(n_chunks, SEQ_LEN))


def process(split: str, tokenizer: Tokenizer, in_dir: Path, out_dir: Path,
            mask_results: bool) -> None:
    examples = load_examples(in_dir / f"{split}.jsonl")
    arr, mask = tokenize_and_pack(examples, tokenizer, mask_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.npy"
    np.save(out_path, arr)
    np.save(out_path.with_suffix(".mask.npy"), mask)

    trained = 100 * mask.mean()
    print(f"{split}: {arr.size:,} tokens, shape {arr.shape} -> {out_path}")
    print(f"  {trained:.1f}% trained on ({100 - trained:.1f}% masked: results and padding)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="data/tokenizer/jaiyu_math_tokenizer.json")
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--train-on-results", action="store_true",
                        help="Also train on calculator results, so the model computes "
                             "them itself instead of calling the tool.")
    args = parser.parse_args()

    if not Path(args.tokenizer).exists():
        raise SystemExit(f"{args.tokenizer} not found; build it with "
                         f"scripts/build_math_tokenizer.py.")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    for split in ("train", "eval"):
        process(split, tokenizer, args.in_dir, args.out_dir, not args.train_on_results)


if __name__ == "__main__":
    main()
