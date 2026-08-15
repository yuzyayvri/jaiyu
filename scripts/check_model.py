import torch

from jaiyu.model.config import GPTConfig
from jaiyu.model.transformer import GPT


def main():
    config = GPTConfig()
    model = GPT(config)
    print(f"parameters: {model.count_parameters():,}")

    idx = torch.randint(0, config.vocab_size, (2, 512))
    logits, loss = model(idx)
    print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
