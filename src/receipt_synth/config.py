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
    """The retail price range of an item kind, in whole currency units, bounds included.

    WHAT "INCLUDED" MEANS HERE, because one caller disagrees. `high` is inclusive as this
    range is *stated* and for two of its three callers in `content_builder`: `_repriced`
    clamps a price to `high`, and `_excluded_ceiling` reports it as attainable. The third,
    `_build_line_item`, draws in ten-kopiyka steps over a half-open interval and so ends
    one step below `high` — the word "inclusive" used to sit here unqualified and read as a
    promise about that draw. Widening the draw to reach the endpoint would move every price
    under every seed to buy an outcome nothing depends on, so the docstring is what moved.

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
def high_frequency_surnames(language: str) -> tuple[str, ...]:
    """The surname pool a personal name is composed from, by document language.

    EMPTY when the language declares none, and that is a meaningful answer rather than a
    missing one: only Ukrainian documents are rendered today, so only `uk` has a narrowed
    set, and the caller falls back to Faker's own pool for the rest. See `personal_names`
    in config/generation.yaml for why the Ukrainian pool is narrowed at all.

    Unlike `placeholder_values`, an absent entry does not raise. A missing vocabulary there
    means a template cannot be printed; here it means the surname is drawn the way it was
    drawn before this pool existed.
    """
    return tuple(load_generation()["personal_names"]["surnames"].get(language, ()))


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
def fiscal_makers(pool: str, country: str) -> tuple[tuple[str, str], ...]:
    """The makers a fiscal receipt may name at its foot, as ``(title_suffix, name)`` pairs.

    `pool` is a top-level block of config/vendors.json — `fiscal_software` for a ПРРО's software
    provider, `fiscal_hardware` for a hardware register's manufacturer — and it is named by
    `receipt.registrars.<kind>.maker_pool` in config/fiscal-rules.yaml rather than chosen here.
    Which pool a kind of register draws from is a fact about the document; the marks themselves
    are data, so they live in the data file and are not restated in the fiscal rules.

    PAIRS, NOT NAMES, because the two are printed forms of ONE fact — which maker produced the
    document. 📄 Line 35 of the form prints the wording «ФІСКАЛЬНИЙ ЧЕК» and then the maker's
    name; 👁 a ПРРО additionally tags the wording with a short abbreviation. Returning them
    together is what stops the paper from showing one provider's tag above another's name.

    `title_suffix` is OPTIONAL and no entry carries one today: 👁 the tag is observed, but no
    published source pairs a tag with a provider, so printing one would assert a pairing nobody
    established. Absence yields ``""`` rather than raising — an entry without a tag is the
    ordinary case, not a gap in the file.
    """
    names = load_vendors()[pool].get(country)
    if not names:
        raise KeyError(f"config/vendors.json lists no {pool!r} entry for {country!r}")
    return tuple(
        (entry.get("title_suffix", ""), entry["display_name"]) for entry in names
    )


@cache
def acquirers(country: str) -> tuple[str, ...]:
    """Card acquirer names as printed on a receipt of this jurisdiction."""
    names = load_vendors()["acquirers"].get(country)
    if not names:
        raise KeyError(f"config/vendors.json lists no acquirer for {country!r}")
    return tuple(names)


@cache
def every_vendor(country: str) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Every vendor of a jurisdiction, across all categories.

    For the ordinary operations on a bank statement, which are 🔴 not a claim's evidence and
    belong to no category at all: a person's account holds payments to whoever that person paid,
    and drawing the noise from the claim's own category would make the relevant row findable by
    its neighbours all being unlike it.

    NOT DE-DUPLICATED, and both halves of that matter. A firm listed under two categories is
    genuinely twice as likely to be paid, and — the half that would be a defect — the entries
    that state NO name are the sole traders whose name is drawn per instance, so collapsing two
    identical ones would collapse two different printed names into one.

    Each vendor comes back as a tuple of sorted key-value pairs rather than a dict, because
    `functools.cache` requires a hashable result and a shared mutable one would let one caller's
    edit reach the next. Keys beginning with `$` are notes to a reader of the file and are
    dropped. Order follows the file, so a seeded draw over the result is reproducible.
    """
    catalogue = load_vendors()["vendors"].get(country)
    if not catalogue:
        raise KeyError(f"config/vendors.json lists no vendor for {country!r}")

    return tuple(
        tuple(sorted((key, value) for key, value in vendor.items() if not key.startswith("$")))
        for vendors in catalogue.values()
        for vendor in vendors
    )


@cache
def banks(country: str) -> tuple[str, ...]:
    """The issuers of a payment confirmation, as printed in its header.

    The `printed_name` of each entry and nothing else: `display_name` names the mark in Latin
    for a reader of config/vendors.json, and `slug` is a key. A bank is drawn per document, so a
    single name would teach a consumer the name instead of the field — the same reason the
    acquirer above stopped being one fixed string.

    Entries keyed by `$` are notes rather than data, and the block is a list here, so nothing
    needs skipping; the guard is that a country with no list raises instead of yielding an empty
    header.
    """
    entries = load_vendors()["banks"].get(country)
    if not entries:
        raise KeyError(f"config/vendors.json lists no bank for {country!r}")
    return tuple(entry["printed_name"] for entry in entries)


# --- generation.yaml: the payment-confirmation draw inputs ---------------------
#
# Its own group rather than more entries above, because every one of them reads the same block
# and the block is a document class. `payment_confirmation` in config/generation.yaml states
# what each share was observed to be.


def _payment_confirmation_generation() -> dict[str, Any]:
    return load_generation()["payment_confirmation"]


@cache
def payment_confirmation_share(name: str) -> float:
    """One of the observed frequencies a payment-confirmation variant is drawn at.

    A single accessor over a family of keys rather than one function per key: they are all the
    same kind of value, they are all read once, and a dozen near-identical loaders would be a
    dozen places to keep in step. The name is checked against the file so a typo fails here,
    naming the key, instead of defaulting to some rate nobody chose.
    """
    block = _payment_confirmation_generation()
    key = f"{name}_share"
    if key not in block:
        raise KeyError(
            f"config/generation.yaml declares no `payment_confirmation.{key}`; it has "
            f"{sorted(k for k in block if k.endswith('_share'))}"
        )
    return float(block[key])


@cache
def payment_confirmation_money_range(name: str) -> tuple[Decimal, Decimal]:
    """A money range of the payment-confirmation block, exact to the kopiyka.

    Parsed from decimal text for the reason `price_range` states: a YAML float is a binary
    double, and these bounds feed money arithmetic that is otherwise exact.
    """
    block = _payment_confirmation_generation()
    key = f"{name}_range"
    if key not in block:
        raise KeyError(f"config/generation.yaml declares no `payment_confirmation.{key}`")
    low, high = (Decimal(str(value)) for value in block[key])
    return low, high


@cache
def initiation_shares() -> dict[str, float]:
    """How a payment was initiated, as weights in the order the file declares them.

    A dict rather than a tuple of pairs because the caller draws by name and the name selects a
    field set out of `payment_confirmation.initiation` in config/fiscal-rules.yaml. Declaration
    order is preserved, which is what keeps a seeded draw reproducible.
    """
    return {
        str(name): float(share)
        for name, share in _payment_confirmation_generation()["initiation_shares"].items()
    }


@cache
def payment_purposes(language: str) -> tuple[str, ...]:
    """The payment-purpose templates of a language.

    🔴 None of them names what was bought — that is the observation `proves_subject: false`
    rests on, and it is stated where the templates are.
    """
    purposes = _payment_confirmation_generation()["purposes"].get(language)
    if not purposes:
        raise KeyError(
            f"config/generation.yaml lists no payment purpose for language {language!r}"
        )
    return tuple(purposes)


@cache
def initiating_systems(language: str) -> tuple[str, ...]:
    """The values a "name of the initiating system" caption may carry."""
    systems = _payment_confirmation_generation()["initiating_systems"].get(language)
    if not systems:
        raise KeyError(
            f"config/generation.yaml lists no initiating system for language {language!r}"
        )
    return tuple(systems)


# --- generation.yaml: the invoice draw inputs ----------------------------------
#
# Third and last per-class group. Three classes now share one shape — a `_share` family, a range
# family — and the third occurrence is where abstracting it becomes right rather than premature.
# It is deliberately NOT abstracted in this commit: the invoice is the change under review, and a
# refactor of the two accessors that already work would put an unrelated diff in front of it.
# Recorded here as the next cleanup rather than left to be noticed.


@cache
def mismatch_delta_range() -> tuple[Decimal, Decimal]:
    """By how much a deliberately mismatched pair disagrees, in hryvnias.

    The SHARE governing how often that happens is in config/policy.yaml beside `verdict_mix`, and
    the split is deliberate: a share sizes a labelled bucket, this magnitude changes no label.
    """
    low, high = (Decimal(str(value)) for value in load_generation()["mismatch"]["delta_range"])
    return low, high


def _invoice_generation() -> dict[str, Any]:
    return load_generation()["invoice"]


@cache
def invoice_share(name: str) -> float:
    """The rate at which an optional requisite of an invoice is printed."""
    block = _invoice_generation()
    key = f"{name}_share"
    if key not in block:
        raise KeyError(
            f"config/generation.yaml declares no `invoice.{key}`; it has "
            f"{sorted(k for k in block if k.endswith('_share'))}"
        )
    return float(block[key])


@cache
def invoice_count_range(name: str) -> tuple[int, int]:
    """An inclusive range of whole things in the invoice block — days of validity."""
    block = _invoice_generation()
    key = f"{name}_range"
    if key not in block:
        raise KeyError(f"config/generation.yaml declares no `invoice.{key}`")
    low, high = (int(value) for value in block[key])
    if low > high:
        raise ValueError(f"config/generation.yaml has `invoice.{key}` reversed: {low}-{high}")
    return low, high


@cache
def phone_prefixes() -> tuple[str, ...]:
    """📄 The mobile prefixes of the national numbering plan.

    A number built from one of these and drawn digits designates nobody; a number copied off a
    document designates whoever holds it, which is why none is.
    """
    return tuple(_invoice_generation()["phone_prefixes"])


# --- generation.yaml: the bank-statement draw inputs ---------------------------
#
# A group of its own, mirroring the confirmation's above, because it reads a different block of
# a different document class. Deliberately NOT folded into one accessor family taking a class
# name: two classes are two, and the shape is worth abstracting on the third rather than on the
# second. What the statement PRINTS is `bank_statement` in config/fiscal-rules.yaml.


def _bank_statement_generation() -> dict[str, Any]:
    return load_generation()["bank_statement"]


@cache
def bank_statement_share(name: str) -> float:
    """One of the observed proportions a bank-statement variant is drawn at."""
    block = _bank_statement_generation()
    key = f"{name}_share"
    if key not in block:
        raise KeyError(
            f"config/generation.yaml declares no `bank_statement.{key}`; it has "
            f"{sorted(k for k in block if k.endswith('_share'))}"
        )
    return float(block[key])


@cache
def bank_statement_money_range(name: str) -> tuple[Decimal, Decimal]:
    """A money range of the bank-statement block, exact to the kopiyka.

    Parsed from decimal text for the reason `price_range` states: a YAML float is a binary
    double, and these bounds feed money arithmetic that is otherwise exact.
    """
    block = _bank_statement_generation()
    key = f"{name}_range"
    if key not in block:
        raise KeyError(f"config/generation.yaml declares no `bank_statement.{key}`")
    low, high = (Decimal(str(value)) for value in block[key])
    return low, high


@cache
def bank_statement_count_range(name: str) -> tuple[int, int]:
    """An inclusive range of whole things — rows on a page, days in a period.

    Kept apart from the money accessor above rather than sharing one that returns `Decimal`: a
    row count is not money, and a caller that received a `Decimal` here would have to convert it
    back before handing it to `random.randint`.
    """
    block = _bank_statement_generation()
    key = f"{name}_range"
    if key not in block:
        raise KeyError(f"config/generation.yaml declares no `bank_statement.{key}`")
    low, high = (int(value) for value in block[key])
    if low > high:
        raise ValueError(
            f"config/generation.yaml has `bank_statement.{key}` reversed: {low}-{high}"
        )
    return low, high


@cache
def statement_purposes(language: str, kind: str) -> tuple[str, ...]:
    """The payment-purpose templates a statement row of this kind may carry.

    `kind` is `debit`, `credit` or `service_fee`, and they are three pools rather than one
    because 🔴 the direction decides what the text can say: a credit row's money is arriving, so
    «Оплата за …» is impossible on it. The service-fee wording is a single string in the file and
    comes back as a one-element tuple, so every caller draws the same way.
    """
    pools = _bank_statement_generation()["purposes"].get(language)
    if not pools:
        raise KeyError(f"config/generation.yaml lists no statement purpose for {language!r}")
    if kind not in pools:
        raise KeyError(
            f"config/generation.yaml lists no `bank_statement.purposes.{language}.{kind}`; "
            f"it has {sorted(pools)}"
        )
    entry = pools[kind]
    return (entry,) if isinstance(entry, str) else tuple(entry)
