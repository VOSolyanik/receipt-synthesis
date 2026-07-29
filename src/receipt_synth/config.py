"""Loading of the configuration files the generator reads.

Kept in its own module because the files are read by stages at opposite ends of the
pipeline — `persona_generator` needs the policy, `renderer` needs the fiscal rules —
and no stage should have to import a downstream one to reach them.

Results are cached: the configuration is read-only input, and reloading it per document
would be pure waste on a dataset of any size. Every cached accessor returns an immutable
value — a tuple or a frozenset — because a cached mutable would let one caller edit the
configuration every later caller sees.
"""

from __future__ import annotations

import json
from decimal import Decimal
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
    jurisdiction, and the item kinds each sort of outlet sells."""
    with (CONFIG_DIR / "vendors.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@cache
def load_generation() -> dict[str, Any]:
    """config/generation.yaml — the vocabulary and distributions the draws use.

    Not policy and not law: what word fills a placeholder, what an article costs, how many
    of it a basket holds, and what coverage ratio a mixed basket aims at. See the head of
    that file for why the coverage targets live there rather than beside `verdict_mix`.
    """
    return _load_yaml("generation.yaml")


def jurisdiction(code: str) -> dict[str, Any]:
    """The fiscal-rules block of one jurisdiction, e.g. ``jurisdiction("UA")``."""
    return load_fiscal_rules()["jurisdictions"][code]


def category(category_id: str) -> dict[str, Any]:
    """One benefit category from the policy, by its id."""
    for entry in load_policy()["categories"]:
        if entry["id"] == category_id:
            return entry
    raise KeyError(f"no such category in policy.yaml: {category_id}")


# --- generation.yaml accessors ------------------------------------------------


@cache
def placeholder_values(item_kind: str, name: str, language: str) -> tuple[str, ...]:
    """What may be substituted for ``{name}`` in a line-item template of ``item_kind``.

    Resolved most specific first: the language's own entry for the kind, then the
    language-neutral one, then the kind-independent defaults in the same order. The
    ordering is what lets `{brand}` mean a laptop maker on `hardware` and a skincare house
    on `cosmetics` while `{period}` is stated once.

    Raises ``KeyError`` when nothing matches, and that is the point. A missing vocabulary
    used to make the builder skip the template silently, which narrowed the printed
    vocabulary of the whole dataset with nothing to show for it — a template that names a
    placeholder nobody has filled is a gap in config/generation.yaml, and it says so here
    rather than disappearing.
    """
    blocks = load_generation()["line_item_placeholders"]
    for scope in (language, "shared"):
        for key in (item_kind, "default"):
            values = blocks.get(scope, {}).get(key, {}).get(name)
            if values is not None:
                return tuple(values)
    raise KeyError(
        f"config/generation.yaml has no vocabulary for placeholder {{{name}}} of item "
        f"kind {item_kind!r} in language {language!r} — add it under "
        f"line_item_placeholders.{language}.{item_kind} (or .shared, or .default)"
    )


@cache
def unprintable_item_kinds() -> frozenset[str]:
    """Item kinds config/generation.yaml declares it cannot print, and why — see the block
    of that name for the reasons.

    A DECLARED hole rather than a silent one. Skipping a template because nobody filled its
    vocabulary is what the loud failure in `placeholder_values` exists to stop; this is the
    opposite — an enumerated, tested statement that a kind cannot be printed, with the cause
    written down. The test suite checks both directions, so an entry that has quietly become
    printable fails as loudly as a kind that is missing one.
    """
    return frozenset(load_generation().get("unprintable_item_kinds", {}))


@cache
def price_range(item_kind: str) -> tuple[Decimal, Decimal]:
    """The inclusive retail price range of an item kind, in whole currency units.

    The same scale `annual_limit` uses in policy.yaml, because the two files are read
    together and a second scale carried only by a field name is a two-order-of-magnitude
    error that prints an implausible receipt and breaks nothing. Whichever unit a draw
    needs is the caller's business.

    Built from decimal text, so the value is exact to the kopiyka; a range stated to more
    places than the currency has is refused rather than rounded, because silently dropping
    a digit of a price is the same class of mistake the scale itself is guarding against.
    """
    block = load_generation()["price_ranges"]
    bounds = []
    for value in block["ranges"].get(item_kind, block["default"]):
        amount = Decimal(str(value))
        if amount != amount.quantize(Decimal("0.01")):
            raise ValueError(
                f"price range of {item_kind!r} in config/generation.yaml states "
                f"{value!r}, which is finer than the currency: at most two decimal places"
            )
        bounds.append(amount)
    low, high = bounds
    return low, high


@cache
def quantity_choices() -> tuple[int, ...]:
    """The pool a line's quantity is drawn from, weighted by repetition."""
    return tuple(load_generation()["basket"]["quantity_choices"])


@cache
def excluded_line_counts() -> tuple[int, ...]:
    """The pool for how many non-covered lines a mixed basket carries."""
    return tuple(load_generation()["basket"]["excluded_line_counts"])


@cache
def coverage_targets() -> tuple[Decimal, ...]:
    """The covered fractions a mixed basket may aim at.

    Parsed from decimal text rather than from YAML floats: 0.35 in YAML is a binary
    double, and these feed money arithmetic that is otherwise exact.
    """
    return tuple(
        Decimal(str(value))
        for value in load_generation()["mixed_basket"]["coverage_targets"]
    )


# --- vendors.json accessors ---------------------------------------------------


@cache
def vendor_profile(slug: str) -> frozenset[str]:
    """The item kinds an outlet of this sort sells.

    A frozenset because it is only ever asked "does it contain this kind" — anything that
    needs an order over item kinds gets it from the catalogue, which is sorted, so no draw
    depends on the iteration order of this value.
    """
    profiles = load_vendors()["vendor_profiles"]
    if slug not in profiles or slug.startswith("$"):
        raise KeyError(f"config/vendors.json declares no vendor profile {slug!r}")
    return frozenset(profiles[slug])


@cache
def acquirers(country: str) -> tuple[str, ...]:
    """Card acquirer names as printed on a receipt of this jurisdiction."""
    names = load_vendors()["acquirers"].get(country)
    if not names:
        raise KeyError(f"config/vendors.json lists no acquirer for {country!r}")
    return tuple(names)
