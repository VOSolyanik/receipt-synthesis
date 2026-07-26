"""Loading of the four configuration files.

Kept in its own module because the files are read by stages at opposite ends of the
pipeline — `persona_generator` needs the policy, `renderer` needs the fiscal rules —
and no stage should have to import a downstream one to reach them.

Results are cached: the configuration is read-only input, and reloading it per document
would be pure waste on a dataset of any size.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@cache
def load_policy() -> dict[str, Any]:
    """config/policy.yaml — the labelling policy. Because generation is label-first,
    this file is the ground truth rather than a description of it."""
    return _load_yaml("policy.yaml")


@cache
def load_fiscal_rules() -> dict[str, Any]:
    """config/fiscal-rules.yaml — law: VAT letters, identifier formats, layout constants."""
    return _load_yaml("fiscal-rules.yaml")


@cache
def load_fx_rates() -> dict[str, Any]:
    """config/fx-rates.yaml — static reference rates. Static so that the same seed
    yields the same labels on any day."""
    return _load_yaml("fx-rates.yaml")


@cache
def load_vendors() -> dict[str, Any]:
    """config/vendors.json — merchant, bank and provider names per category and
    jurisdiction."""
    with (CONFIG_DIR / "vendors.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def jurisdiction(code: str) -> dict[str, Any]:
    """The fiscal-rules block of one jurisdiction, e.g. ``jurisdiction("UA")``."""
    return load_fiscal_rules()["jurisdictions"][code]


def category(category_id: str) -> dict[str, Any]:
    """One benefit category from the policy, by its id."""
    for entry in load_policy()["categories"]:
        if entry["id"] == category_id:
            return entry
    raise KeyError(f"no such category in policy.yaml: {category_id}")
