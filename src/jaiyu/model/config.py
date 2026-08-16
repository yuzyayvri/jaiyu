from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/model/jaiyu_26m.yaml")


@dataclass
class GPTConfig:
    vocab_size: int = 413
    block_size: int = 512
    n_layer: int = 14
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.0


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> GPTConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return GPTConfig(**data)
