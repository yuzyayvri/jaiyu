from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 224
    dropout: float = 0.0
