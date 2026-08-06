# One command that means "I ran the project's checks", so that phrase means the same
# thing to every contributor.
#
# Every recipe goes through `uv run`, which resolves the project's own .venv. The empty
# VIRTUAL_ENV= prefix mutes uv's warning when an unrelated virtualenv is active in the
# caller's shell. Never `uv run --active`: it once synced this project into an unrelated
# virtualenv and uninstalled another project's packages.
UV := VIRTUAL_ENV= uv run

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  check      Run lint, tests and the redaction gate — the gate before calling anything done"
	@echo "  lint       Run ruff"
	@echo "  test       Run pytest"
	@echo "  redaction  Run the redaction gate (fails if an unpublishable term is in the tree)"
	@echo "  cross-seed Assert that a corpus is a function of its seed and of nothing else"
	@echo "  install    Sync dependencies and install the Chromium build the renderer needs"

# Ordered cheapest-first: a lint error is reported without waiting for the full test run.
# .NOTPARALLEL keeps that order (and the stop-at-first-failure it buys) under `make -j`.
.NOTPARALLEL:

.PHONY: check
check: lint test redaction

.PHONY: lint
lint:
	$(UV) ruff check .

.PHONY: test
test:
	$(UV) pytest

.PHONY: redaction
redaction:
	$(UV) python tools/check_redaction.py

# Deliberately not part of `check`: it generates three small corpora, which costs about half a
# minute, and `check` is the target people run between edits. CI runs it on every push instead.
.PHONY: cross-seed
cross-seed:
	$(UV) python tools/cross_seed_check.py

# Chromium is not a Python dependency, so `uv sync` alone leaves the renderer unable to run.
.PHONY: install
install:
	VIRTUAL_ENV= uv sync
	$(UV) playwright install chromium
