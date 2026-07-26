"""The amount printed in words must state the same number as the amount in digits.

The check is non-circular on purpose. The formatter is pinned by hand-written expected
strings, which are the ground truth for Ukrainian declension; the parser is an
independent reading of the text; and the round trip catches anything ambiguous or
lossy. A validator that merely re-ran the formatter and compared would confirm nothing.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from receipt_synth.content_builder import (
    amount_in_words_uk,
    validate_amount_in_words,
    words_to_amount_uk,
)


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        # Hryvnia and kopiyka are both feminine, so the numeral is feminine too:
        # "одна", "дві" — never "один", "два".
        ("1.00", "одна гривня 00 копійок"),
        ("2.00", "дві гривні 00 копійок"),
        ("5.00", "п'ять гривень 00 копійок"),
        # 11–14 take the genitive plural despite ending in 1–4.
        ("11.00", "одинадцять гривень 00 копійок"),
        ("12.00", "дванадцять гривень 00 копійок"),
        ("14.00", "чотирнадцять гривень 00 копійок"),
        # …while 21, 22 follow their last digit again.
        ("21.00", "двадцять одна гривня 00 копійок"),
        ("22.00", "двадцять дві гривні 00 копійок"),
        ("25.00", "двадцять п'ять гривень 00 копійок"),
        # The kopiyka word declines by the kopiyka count, independently of hryvnias.
        ("10.01", "десять гривень 01 копійка"),
        ("10.02", "десять гривень 02 копійки"),
        ("10.05", "десять гривень 05 копійок"),
        ("10.11", "десять гривень 11 копійок"),
        ("10.21", "десять гривень 21 копійка"),
        # Hundreds and the feminine thousand.
        ("100.00", "сто гривень 00 копійок"),
        ("247.35", "двісті сорок сім гривень 35 копійок"),
        ("1000.00", "одна тисяча гривень 00 копійок"),
        ("2000.00", "дві тисячі гривень 00 копійок"),
        ("5000.00", "п'ять тисяч гривень 00 копійок"),
        ("2500.00", "дві тисячі п'ятсот гривень 00 копійок"),
        ("11000.00", "одинадцять тисяч гривень 00 копійок"),
        ("21000.00", "двадцять одна тисяча гривень 00 копійок"),
        ("999999.99", "дев'ятсот дев'яносто дев'ять тисяч дев'ятсот дев'яносто дев'ять "
                      "гривень 99 копійок"),
        # Million is masculine — "один мільйон", not "одна".
        ("1000000.00", "один мільйон гривень 00 копійок"),
        ("2000000.00", "два мільйони гривень 00 копійок"),
        ("0.50", "нуль гривень 50 копійок"),
    ],
)
def test_formatter_matches_expected_ukrainian(amount, expected):
    assert amount_in_words_uk(Decimal(amount)) == expected


@pytest.mark.parametrize(
    "amount",
    ["0.00", "0.01", "1.01", "3.33", "19.99", "111.11", "1234.56", "10000.00", "987654.32"],
)
def test_round_trip(amount):
    value = Decimal(amount)
    assert words_to_amount_uk(amount_in_words_uk(value)) == value


def test_round_trip_over_a_deterministic_sweep():
    """A sweep wide enough to hit every declension branch of both units."""
    for kopiykas in range(0, 200_000, 137):
        value = (Decimal(kopiykas) / 100).quantize(Decimal("0.01"))
        assert words_to_amount_uk(amount_in_words_uk(value)) == value


def test_validator_accepts_matching_text():
    assert validate_amount_in_words("дві тисячі п'ятсот гривень 00 копійок", Decimal("2500.00"))


def test_validator_rejects_a_different_number():
    """The failure a fraud archetype is built on: the digits and the words disagree."""
    assert not validate_amount_in_words(
        "дві тисячі п'ятсот гривень 00 копійок", Decimal("2600.00")
    )


def test_validator_rejects_a_kopiyka_mismatch():
    assert not validate_amount_in_words("десять гривень 05 копійок", Decimal("10.50"))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "дві тисячі п'ятсот",  # no unit words
        "дві тисячі п'ятсот гривень",  # kopiykas missing
        "two thousand five hundred hryvnias 00 kopiykas",
        "мільярд гривень 00 копійок",  # a numeral the generator never emits
    ],
)
def test_parser_rejects_unreadable_text(text):
    with pytest.raises(ValueError):
        words_to_amount_uk(text)


def test_validator_is_false_rather_than_raising_on_unreadable_text():
    """A trap archetype may print nonsense where the amount in words belongs. The
    validator answers the question asked — "does this text state this amount" — and an
    unreadable string answers it with no."""
    assert not validate_amount_in_words("???", Decimal("10.00"))


def test_formatter_rejects_more_than_two_decimal_places():
    with pytest.raises(ValueError):
        amount_in_words_uk(Decimal("10.005"))


def test_formatter_rejects_a_negative_amount():
    with pytest.raises(ValueError):
        amount_in_words_uk(Decimal("-1.00"))
