"""ПДВ-літера ↔ rate, per config/fiscal-rules.yaml.

The mapping is read from the configuration rather than restated here: a test that
hardcoded it would pass while the rendered receipts used something else.
"""

from __future__ import annotations

import random

import pytest

from receipt_synth.config import jurisdiction
from receipt_synth.content_builder import (
    validate_vat_letter,
    vat_letter_for_kind,
    vat_rate_for_letter,
)

UA = jurisdiction("UA")


def test_rates_come_from_the_configuration():
    for letter, spec in UA["vat_letters"].items():
        assert vat_rate_for_letter(letter, "UA") == spec["rate"]


def test_unknown_letter_is_an_error():
    with pytest.raises(KeyError):
        vat_rate_for_letter("Я", "UA")


def test_single_valued_kinds_are_fixed():
    """Medicines take the reduced 7% group, supplements the standard 20% one. Neither
    depends on the seed."""
    rng = random.Random(1)
    assert vat_letter_for_kind("medicine", "UA", rng) == "В"
    assert vat_letter_for_kind("mineral_supplement", "UA", rng) == "А"
    assert vat_rate_for_letter("В", "UA") == 7.0
    assert vat_rate_for_letter("А", "UA") == 20.0


def test_unknown_kind_falls_back_to_the_configured_default():
    rng = random.Random(1)
    assert vat_letter_for_kind("no_such_kind", "UA", rng) == UA["item_vat_letter"]["default"]


def test_vitamin_complex_draws_from_the_configured_list():
    """A vitamin complex may be registered as a medicinal product or as a dietary
    supplement, and real receipts show both — so the letter is drawn, not fixed."""
    allowed = set(UA["item_vat_letter"]["vitamin_complex"])
    assert allowed == {"В", "А"}
    drawn = {vat_letter_for_kind("vitamin_complex", "UA", random.Random(s)) for s in range(50)}
    assert drawn == allowed, "both letters must be reachable, or the list is decorative"


def test_vitamin_complex_choice_is_deterministic_under_seed():
    for seed in range(20):
        first = vat_letter_for_kind("vitamin_complex", "UA", random.Random(seed))
        second = vat_letter_for_kind("vitamin_complex", "UA", random.Random(seed))
        assert first == second


def test_the_letter_is_not_a_shortcut_to_the_coverage_label():
    """The property policy.yaml is built around: a covered vitamin complex can carry the
    same letter as a non-covered medicine. If this ever became false, a consumer could
    read coverage off the VAT letter without reading the line item, and the dataset
    would stop testing what it exists to test."""
    vitamin_letters = set(UA["item_vat_letter"]["vitamin_complex"])
    assert UA["item_vat_letter"]["medicine"] in vitamin_letters


def test_validator_accepts_a_letter_the_kind_allows():
    assert validate_vat_letter("vitamin_complex", "В", "UA")
    assert validate_vat_letter("vitamin_complex", "А", "UA")
    assert validate_vat_letter("medicine", "В", "UA")


def test_validator_rejects_a_letter_the_kind_does_not_allow():
    """Medicine at 20% is a broken invariant. A fraud archetype may print it — but only
    as a labelled choice, never because the builder drifted."""
    assert not validate_vat_letter("medicine", "А", "UA")
    assert not validate_vat_letter("mineral_supplement", "В", "UA")


def test_validator_rejects_a_letter_outside_the_alphabet():
    assert not validate_vat_letter("vitamin_complex", "Z", "UA")
