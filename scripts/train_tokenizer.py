import argparse
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/intermediate/synthetic/train.jsonl")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    out_path = "data/tokenizer/jaiyu_tokenizer.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def text_iterator():
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                yield example["text"] + "Answer: " + example["answer"] + "\n"

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Digits first: without it BPE merges whole numbers into atomic symbols
    # ("34" and "49" become unrelated ids), so the model can only memorise an
    # a+b lookup table instead of learning digit arithmetic. Splitting to
    # individual digits gives it 10 reusable symbols with place structure.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<eos>", "<unk>"],
        # Seed the full byte alphabet so a character the corpus gains later
        # (e.g. "quotient" after a generator change) can never become <unk>.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(out_path)

    vocab = tokenizer.get_vocab()
    print(f"Vocab size: {len(vocab)}")
    sorted_tokens = sorted(vocab.items(), key=lambda kv: kv[1])
    for token, idx in sorted_tokens[:30]:
        print(idx, token)


if __name__ == "__main__":
    main()
