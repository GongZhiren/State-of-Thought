PYTHON ?= python
CONFIG ?= experiments/paper/llama-3.1-8b.yaml

.PHONY: install check test verify data-check evaluate train

install:
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m ruff check src scripts tests
	PYTHONPATH=src $(PYTHON) scripts/check_release.py
	PYTHONPATH=src $(PYTHON) scripts/verify_results.py
	PYTHONPATH=src $(PYTHON) scripts/build_result_summaries.py --check
	$(PYTHON) -m compileall -q src scripts tests

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

verify:
	PYTHONPATH=src $(PYTHON) -m sot.cli checkpoint verify checkpoints/llama-3.1-8b/controller.pt

data-check:
	PYTHONPATH=src $(PYTHON) -m sot.cli data validate --config $(CONFIG)

evaluate:
	PYTHONPATH=src $(PYTHON) -m sot.cli evaluate --config $(CONFIG)

train:
	PYTHONPATH=src $(PYTHON) -m sot.cli train --config $(CONFIG)
