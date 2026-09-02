.PHONY: help data demo test install clean

help:
	@echo "make data     - regenerate the synthetic corpus (data/*.json)"
	@echo "make demo     - run the six end-to-end scenarios"
	@echo "make test     - run the pytest suite"
	@echo "make install  - pip install -e . (editable)"
	@echo "make clean    - remove caches"

data:
	python3 data/generate_dataset.py

demo:
	python3 examples/demo.py

test:
	python3 -m pytest

install:
	python3 -m pip install -e ".[dev]"

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info build dist
