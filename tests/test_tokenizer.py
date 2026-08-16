"""Guards against the two tokenizer regressions that silently broke arithmetic.

1. Whole numbers merged into atomic tokens ("34" as one id), which leaves the
   model no digit structure to generalise over.
2. A character missing from the vocabulary becoming <unk> after a generator
   change added a new word (this happened to "quotient").
"""

import json
from pathlib import Path

import pytest

Tokenizer = pytest.importorskip("tokenizers").Tokenizer

TOKENIZER_PATH = Path("data/tokenizer/jaiyu_tokenizer.json")
TRAIN_PATH = Path("data/intermediate/synthetic/train.jsonl")

pytestmark = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="tokenizer not trained yet"
)


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@pytest.mark.parametrize("number", ["0", "7", "34", "144", "8148"])
def test_numbers_split_into_single_digits(tokenizer, number):
    tokens = [t.lstrip("Ġ") for t in tokenizer.encode(number).tokens]
    assert [t for t in tokens if t] == list(number)


def test_training_corpus_has_no_unknown_tokens(tokenizer):
    if not TRAIN_PATH.exists():
        pytest.skip("no synthetic data generated yet")
    unk_id = tokenizer.token_to_id("<unk>")
    with TRAIN_PATH.open() as f:
        for line in f:
            example = json.loads(line)
            text = example["text"] + "Answer: " + example["answer"] + "\n"
            assert unk_id not in tokenizer.encode(text).ids, f"<unk> in: {text!r}"
