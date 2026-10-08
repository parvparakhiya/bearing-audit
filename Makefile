.PHONY: install fetch verify physics audit industrial figures all notebook test lint typecheck check docker clean

install:            ## pinned versions (the ones that produced reports/), then editable install
	pip install -r requirements-lock.txt && pip install -e ".[dev]"

fetch:              ## download the 56 recordings, verify SHA-256 against data/manifest.csv
	bearing-audit fetch

verify:             ## checksums, sampling-rate proof, acquisition-confound check
	bearing-audit verify

physics:            ## fault-frequency verification, every faulted recording
	bearing-audit physics

audit:              ## features + three split protocols + pre-registered gate
	bearing-audit audit

industrial:         ## plant-level evaluation: detection, three-state diagnosis, cost, gate
	bearing-audit industrial

figures:            ## render reports/figures from reports/results
	bearing-audit figures

all:                ## the full pipeline, from verified data to figures
	bearing-audit all

notebook:           ## re-execute both notebooks in place (needs .[notebook] and the data); every check must pass
	jupyter nbconvert --to notebook --execute --inplace notebooks/bearing_audit_presentation.ipynb
	BEARING_AUDIT_STRICT=1 jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/bearing_audit_full_pipeline.ipynb

test:
	pytest -rs

lint:
	ruff check src tests notebooks && ruff format --check src tests notebooks

typecheck:
	python -m mypy

check: lint typecheck test   ## everything CI runs without the data

docker:             ## build the image (pinned versions)
	docker build -t bearing-audit:1.0.0 .

clean:              ## remove generated results (figures and data are kept)
	rm -f reports/results/features.csv
