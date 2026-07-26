"""Checksum invariants for Ukrainian taxpayer identifiers.

Both algorithms are exercised in three directions: a generated identifier validates, a
tampered one does not, and generation is reproducible under a seed.
"""

from __future__ import annotations

import random
from datetime import date

import pytest

from receipt_synth.content_builder import (
    edrpou_check_digit,
    generate_edrpou,
    generate_rnokpp,
    is_valid_edrpou,
    is_valid_rnokpp,
    rnokpp_check_digit,
)

# --------------------------------------------------------------- РНОКПП (10) --


def test_rnokpp_generated_is_valid():
    rng = random.Random(7)
    for _ in range(200):
        assert is_valid_rnokpp(generate_rnokpp(rng))


def test_rnokpp_has_ten_digits():
    rng = random.Random(7)
    value = generate_rnokpp(rng)
    assert len(value) == 10
    assert value.isdigit()


def test_rnokpp_encodes_the_birth_date():
    """The first five digits are the number of days from 1899-12-31 to the birth date.

    1900-01-01 is day 1, which pins the epoch: an off-by-one here would produce
    identifiers that pass the checksum but disagree with the persona's age.
    """
    rng = random.Random(1)
    value = generate_rnokpp(rng, birth_date=date(1900, 1, 1))
    assert value[:5] == "00001"

    value = generate_rnokpp(rng, birth_date=date(1990, 5, 17))
    assert value[:5] == "33009"


def test_rnokpp_rejects_a_tampered_check_digit():
    rng = random.Random(3)
    value = generate_rnokpp(rng)
    wrong = value[:9] + str((int(value[9]) + 1) % 10)
    assert not is_valid_rnokpp(wrong)


def test_rnokpp_rejects_a_tampered_body():
    """A changed body digit must invalidate the identifier, not merely change it."""
    rng = random.Random(4)
    value = generate_rnokpp(rng)
    # Weight of position 3 is 7, so a single-step change cannot be absorbed mod 11.
    tampered = value[:2] + str((int(value[2]) + 1) % 10) + value[3:]
    assert tampered != value
    assert not is_valid_rnokpp(tampered)


@pytest.mark.parametrize("bad", ["", "123", "12345678901", "abcdefghij", "123456789 "])
def test_rnokpp_rejects_malformed_input(bad):
    assert not is_valid_rnokpp(bad)


def test_rnokpp_is_deterministic_under_seed():
    assert generate_rnokpp(random.Random(42)) == generate_rnokpp(random.Random(42))


def test_rnokpp_check_digit_matches_the_published_weights():
    """Weights (-1, 5, 7, 9, 4, 6, 10, 5, 7), check = sum mod 11 mod 10."""
    weights = (-1, 5, 7, 9, 4, 6, 10, 5, 7)
    for body in ("000000000", "123456789", "300000001", "987654321"):
        expected = sum(w * int(d) for w, d in zip(weights, body, strict=True)) % 11 % 10
        assert rnokpp_check_digit(body) == expected


# ---------------------------------------------------------------- ЄДРПОУ (8) --


def test_edrpou_generated_is_valid():
    rng = random.Random(11)
    for _ in range(200):
        assert is_valid_edrpou(generate_edrpou(rng))


def test_edrpou_has_eight_digits():
    rng = random.Random(11)
    value = generate_edrpou(rng)
    assert len(value) == 8
    assert value.isdigit()


def test_edrpou_rejects_a_tampered_check_digit():
    rng = random.Random(5)
    value = generate_edrpou(rng)
    wrong = value[:7] + str((int(value[7]) + 1) % 10)
    assert not is_valid_edrpou(wrong)


@pytest.mark.parametrize("bad", ["", "1234567", "123456789", "abcdefgh"])
def test_edrpou_rejects_malformed_input(bad):
    assert not is_valid_edrpou(bad)


def test_edrpou_is_deterministic_under_seed():
    assert generate_edrpou(random.Random(42)) == generate_edrpou(random.Random(42))


def test_edrpou_switches_weight_set_by_range():
    """Codes in 30000000–60000000 take the rotated weight set (7,1,2,3,4,5,6); all
    others take (1,2,3,4,5,6,7). The two sets disagree for most prefixes, so a code
    valid under one is almost never valid under the other."""
    low = (1, 2, 3, 4, 5, 6, 7)
    mid = (7, 1, 2, 3, 4, 5, 6)

    def first_pass(prefix: str, weights: tuple[int, ...]) -> int:
        return sum(w * int(d) for w, d in zip(weights, prefix, strict=True)) % 11

    assert edrpou_check_digit("12345670") == first_pass("1234567", low)
    assert edrpou_check_digit("32345670") == first_pass("3234567", mid)


def test_edrpou_reports_prefixes_that_carry_no_check_digit():
    """When both passes yield 10 the code is not issuable. That must surface as None,
    not silently as 0 — a fabricated 0 would produce identifiers that no register
    would ever contain."""
    # Step by 10: the check digit is a property of the first seven digits, so
    # consecutive codes share a prefix and scanning codes would sample 10× less widely
    # than it looks. Roughly 1% of prefixes are unissuable.
    unissuable = [
        f"{n:08d}"
        for n in range(10_000_000, 10_020_000, 10)
        if edrpou_check_digit(f"{n:08d}") is None
    ]
    assert unissuable, "expected at least one prefix with no valid check digit"
    for code in unissuable:
        assert not is_valid_edrpou(code)
