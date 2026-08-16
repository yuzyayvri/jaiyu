# Jaiyu

Personal, non-commercial research project: grow a ~26M-parameter math-focused
language model from scratch, locally, and observe how far it can be pushed
in mathematical reasoning.


## Hardware

- AMD Radeon RX 6600, 8 GB VRAM
- Arch Linux, ROCm
- ~150 GB local storage

If ROCm doesn't detect the GPU correctly, try:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Layout

- `configs/` — model, data, training, eval configs (YAML)
- `data/` — raw, processed, and held-out eval data (not committed)
- `src/jaiyu/` — library code (model, data, training, eval)
- `scripts/` — entry-point CLI scripts
- `tests/` — unit tests
- `notebooks/` — exploratory analysis
- `outputs/` — checkpoints, logs (not committed)

## Quickstart (TODO)

```bash
make gpu-check
make data-synth
make tokenize
make train
make eval
```

## License

MIT. See `LICENSE`.
