#!/usr/bin/env python
"""Evaluate a checkpoint on held-out eval data.

TODO: load configs/eval.yaml, load checkpoint (src/jaiyu/model.py).
TODO: run held-out eval sets, compute metrics (src/jaiyu/eval.py).
TODO: print/save simple, interpretable results — per topic/difficulty.
"""

from jaiyu import evaluate  # noqa: F401  TODO: use jaiyu.evaluate eval loop


def main() -> None:
    raise NotImplementedError("TODO: implement evaluation entry point")


if __name__ == "__main__":
    main()
