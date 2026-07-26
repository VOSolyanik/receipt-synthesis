"""Σ(qty × price) over the line items must equal the receipt total, to the kopiyka.

Strict equality, not a tolerance. Amounts are Decimal precisely so that no tolerance is
needed, and a tolerance here would hide the very drift the invariant exists to catch.
"""

from __future__ import annotations

from decimal import Decimal

from receipt_synth.content_builder import line_items_total, validate_line_item_sum
from receipt_synth.schemas import LineItem


def item(qty: str, price: str) -> LineItem:
    return LineItem(
        name="x",
        item_kind="vitamin_complex",
        qty=Decimal(qty),
        price=Decimal(price),
        covered=True,
        vat_letter="В",
    )


def test_total_of_a_single_line():
    assert line_items_total([item("1", "249.90")]) == Decimal("249.90")


def test_price_is_the_unit_price_not_the_line_total():
    """Three packs at 249.90 is 749.70. If `price` were ever read as the line total this
    would come out at 249.90 and every multi-quantity receipt would be wrong."""
    assert line_items_total([item("3", "249.90")]) == Decimal("749.70")


def test_total_of_several_lines():
    items = [item("1", "249.90"), item("2", "112.55"), item("1", "38.00")]
    assert line_items_total(items) == Decimal("513.00")


def test_fractional_quantity():
    """Weighed goods carry a fractional quantity; the line still resolves to kopiykas."""
    assert line_items_total([item("0.5", "100.00")]) == Decimal("50.00")


def test_total_is_quantized_to_two_places():
    total = line_items_total([item("3", "0.33")])
    assert total == Decimal("0.99")
    assert total.as_tuple().exponent == -2


def test_validator_accepts_the_matching_total():
    items = [item("1", "249.90"), item("2", "112.55")]
    assert validate_line_item_sum(items, Decimal("475.00"))


def test_validator_rejects_a_one_kopiyka_discrepancy():
    """One kopiyka is the whole point: an approximate check would pass this."""
    items = [item("1", "249.90"), item("2", "112.55")]
    assert not validate_line_item_sum(items, Decimal("475.01"))
    assert not validate_line_item_sum(items, Decimal("474.99"))


def test_validator_rejects_an_empty_receipt():
    """A receipt with no lines and a non-zero total states an amount nothing accounts
    for — the shape of a document that does not prove what was bought."""
    assert not validate_line_item_sum([], Decimal("100.00"))


def test_validator_accepts_a_total_written_with_trailing_zeros():
    """Decimal("475.0") and Decimal("475.00") are equal but not identical; the check
    must compare numbers, not their representation."""
    items = [item("1", "249.90"), item("2", "112.55")]
    assert validate_line_item_sum(items, Decimal("475.0"))


def test_summation_does_not_accumulate_float_error():
    """The failure mode Decimal is here to prevent: 0.1 + 0.2 in binary floating point
    is not 0.3, and a hundred such lines drift visibly."""
    items = [item("1", "0.10"), item("1", "0.20")]
    assert validate_line_item_sum(items, Decimal("0.30"))
    assert line_items_total([item("1", "0.07")] * 100) == Decimal("7.00")
