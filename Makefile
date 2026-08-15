export HSA_OVERRIDE_GFX_VERSION := 10.3.0
export HCC_AMDGPU_TARGET := gfx1030

.PHONY: gpu-check data-synth data-download tokenize train eval sample test

gpu-check:
	python scripts/check_gpu.py

data-synth:
	python scripts/make_synthetic_data.py

data-download:
	python scripts/download_public_data.py

tokenize:
	python scripts/tokenize_data.py

train:
	python scripts/train.py

eval:
	python scripts/evaluate.py

sample:
	python scripts/sample.py

test:
	pytest tests/
