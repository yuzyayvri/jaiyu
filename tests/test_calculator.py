import types
from pathlib import Path

from jaiyu.calculator import _self_check

TOKENIZER = Path("data/tokenizer/jaiyu_tokenizer.json")


def test_calculator_self_check():
    _self_check()


def test_generation_feeds_results_back():
    # The tokenizer is a build artifact, not in the repo; skip if absent.
    if not TOKENIZER.exists():
        print("skip: no tokenizer at", TOKENIZER)
        return
    import sys

    import torch
    from tokenizers import Tokenizer

    sys.path.insert(0, "scripts")
    import inference_compute as ic

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    script = tokenizer.encode(" Add the ones: <calc> 4 5 + 1 5 </calc>").ids
    script += tokenizer.encode(". Answer: 6 0\n\n").ids

    class Fake:
        """Emits `script` whatever the input; ignores the fed-back result."""

        def __init__(self):
            self.i = 0

        def __call__(self, idx):
            logits = torch.full((1, idx.shape[1], tokenizer.get_vocab_size()), -1e9)
            logits[0, -1, script[min(self.i, len(script) - 1)]] = 0.0
            self.i += 1
            return logits, None

    _, completion = ic.generate(
        Fake(),
        types.SimpleNamespace(block_size=512),
        tokenizer,
        "Question: What is 45 + 15?\nThought:",
        torch.device("cpu"),
        seed=0,
        temperature=0.0,
        max_new_tokens=40,
    )
    assert "<calc> 4 5 + 1 5 </calc> <result> 6 0 </result>" in completion, completion


if __name__ == "__main__":
    test_calculator_self_check()
    test_generation_feeds_results_back()
    print("ok")
