#!/usr/bin/env python
"""Train the 12k-token math BPE tokenizer used for Jaiyu pretraining v2."""

import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 12_000
SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<calc>", "</calc>", "<result>", "</result>",
    "<code>", "</code>", "<output>", "</output>",
]

# The synthetic corpus alone only yields ~600 merges (it is a handful of
# templates over digits), so the external corpora are trained on too.
CORPORA = [Path("data/pretrain/math_corpus.jsonl"), Path("data/pretrain/external")]
OUT_PATH = Path("data/tokenizer/jaiyu_math_tokenizer.json")

_DIGIT = re.compile(r"(\d)")
_SPACES = re.compile(r"[ \t]+")


def split_digits(text: str) -> str:
    """Space out every digit so BPE can never merge numbers. Idempotent."""
    return _SPACES.sub(" ", _DIGIT.sub(r" \1 ", text))


def jsonl_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(
                f"{p} not found. Training on a partial corpus silently produces a "
                f"tokenizer that does not match the data. Generated corpora are not "
                f"in git; run scripts/make_pretrain_data.py and "
                f"scripts/download_math_data.py first."
            )
    assert files, "no training corpora found"
    return files


def corpus_iter(paths: list[Path], limit: int | None):
    for path in jsonl_files(paths):
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                if line.strip():
                    yield split_digits(json.loads(line)["text"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, nargs="+", default=CORPORA,
                   help="jsonl files or directories of jsonl files to train on.")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--smoke-test", action="store_true", help="Train on the first 1000 examples only.")
    args = p.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Digits pre-tokenizer as well as the regex: the regex protects the merge
    # table, this protects text that reaches the tokenizer unsplit at inference.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    # Specials are added after training, so BPE gets vocab_size - len(specials)
    # and the final vocabulary lands on exactly VOCAB_SIZE.
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE - len(SPECIAL_TOKENS),
        min_frequency=2,
        show_progress=True,
        # Seed the whole byte alphabet so unseen characters never become <unk>.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    limit = 1000 if args.smoke_test else None
    tokenizer.train_from_iterator(corpus_iter(args.corpus, limit), trainer=trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))

    vocab = tokenizer.get_vocab()
    print(f"saved {args.out}")
    print(f"vocab size: {len(vocab)} (target {VOCAB_SIZE})")

    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False).ids
        assert len(ids) == 1, f"digit {d} is not a single token: {ids}"
    print("digits 0-9: all single tokens")

    for tok in SPECIAL_TOKENS:
        assert tok in vocab, f"missing special token {tok}"
    print(f"special tokens present: {' '.join(SPECIAL_TOKENS)}")

    for sample in [
        "34 + 15 = 49",
        "Solve 4x - 8 = 12",
        "<calc> 34 + 15 </calc> <result> 49 </result>",
    ]:
        enc = tokenizer.encode(split_digits(sample))
        print(f"\n{sample!r}\n  ids:    {enc.ids}\n  tokens: {enc.tokens}")


if __name__ == "__main__":
    main()
