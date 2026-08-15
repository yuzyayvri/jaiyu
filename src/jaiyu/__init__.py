"""Jaiyu: a ~26M-parameter math-focused language model, trained locally."""

import os

# The RX 6600 is gfx1032, which the ROCm PyTorch wheels do not ship kernels for.
# Without this override every GPU kernel launch fails with
# "HIP error: invalid device function" or aborts the HSA queue with
# HSA_STATUS_ERROR_EXCEPTION. gfx1032 is binary-compatible with gfx1030.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

__version__ = "0.1.0"
