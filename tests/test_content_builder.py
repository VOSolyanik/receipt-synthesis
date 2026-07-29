"""The assembled ПРРО receipt — every invariant holding at once, on generated content.

The per-invariant suites pin each rule in isolation. This one asserts that the builder
actually respects all of them together, over many seeds, which is the only way a drift
between two individually-correct pieces shows up.
"""

from __future__ import annotations

import random
import re
from datetime import date, datetime
from decimal import Decimal

import pytest

from receipt_synth.config import category, jurisdiction
from receipt_synth.content_builder import (
    MAX_LINE_ITEMS,
    build_prro_receipt,
    estimated_line_value,
    is_valid_edrpou,
    renderable_kinds,
    validate_amount_in_words,
    validate_line_item_sum,
    validate_vat_letter,
)
from receipt_synth.policy_engine import covered_total, resolved_coverage, verdict_for
from receipt_synth.schemas import Capture, DocType, Verdict

ISSUED_AT = datetime(2026, 8, 3, 14, 22, 51)
VENDOR = {"name": "Аптека АНЦ", "legal_form": "TOV"}


def build(seed: int, **kwargs):
    return build_prro_receipt(
        random.Random(seed),
        category_id="vitamins_nutrition",
        issued_at=ISSUED_AT,
        vendor=VENDOR,
        **kwargs,
    )


# ------------------------------------------------------------------ contract --


def test_is_deterministic_under_seed():
    assert build(42) == build(42)


def test_different_seeds_give_different_receipts():
    """If they did not, the seed would not be reaching the content and a whole dataset
    would be one receipt repeated."""
    assert len({build(s).receipt_number for s in range(20)}) > 1
    assert len({build(s).total for s in range(20)}) > 1


def test_has_at_least_two_line_items():
    for seed in range(20):
        assert len(build(seed).line_items) >= 2


# ---------------------------------------------------------------- invariants --


@pytest.mark.parametrize("seed", range(30))
def test_all_invariants_hold(seed):
    receipt = build(seed)

    assert validate_line_item_sum(receipt.line_items, receipt.total)
    assert validate_amount_in_words(receipt.amount_in_words, receipt.total)
    # A ТОВ seller is identified by its ЄДРПОУ, printed as "ІД"; a ФОП seller would
    # carry a РНОКПП printed as "ІПН", which is why the field is not named after either.
    assert receipt.seller.tax_code_label == "ІД"
    assert is_valid_edrpou(receipt.seller.tax_code)
    for item in receipt.line_items:
        assert validate_vat_letter(item.item_kind, item.vat_letter, "UA")


@pytest.mark.parametrize("seed", range(30))
def test_covered_only_receipt_contains_no_excluded_item(seed):
    """The label-first knob for this skeleton: the planner asked for `covered`, so the
    builder may draw only from the category's covered items. A single excluded line
    would silently make the claim partially covered and the label wrong."""
    covered_kinds = set(category("vitamins_nutrition")["covered_items"])
    receipt = build(seed, covered_only=True)

    assert all(item.covered for item in receipt.line_items)
    assert {item.item_kind for item in receipt.line_items} <= covered_kinds


# ---------------------------------------------------------- mixed baskets ----


def mixed(seed: int, coverage_target: str = "0.7", **kwargs):
    return build(
        seed, covered_only=False, coverage_target=Decimal(coverage_target), **kwargs
    )


@pytest.mark.parametrize("seed", range(30))
def test_a_mixed_basket_draws_from_both_buckets(seed):
    spec = category("vitamins_nutrition")
    receipt = mixed(seed)

    kinds = {item.item_kind for item in receipt.line_items}
    assert kinds & set(spec["covered_items"]), "no covered line"
    assert kinds & set(spec["excluded_items"]), "no non-covered line"
    assert kinds.isdisjoint(spec["ambiguous_items"]), (
        "policy.yaml states no coverage answer for the ambiguous bucket, so a line drawn "
        "from it would carry a `covered` flag this generator invented"
    )


@pytest.mark.parametrize("seed", range(30))
def test_the_covered_flag_agrees_with_the_bucket_the_line_came_from(seed):
    """The flag is the label. A line from `excluded_items` marked covered would be a
    wrong per-line label *and* a wrong claim verdict, from one mistake."""
    spec = category("vitamins_nutrition")
    for item in mixed(seed).line_items:
        assert item.covered is (item.item_kind in spec["covered_items"])
        assert item.covered is not (item.item_kind in spec["excluded_items"])


@pytest.mark.parametrize("seed", range(40))
def test_a_mixed_basket_lands_on_the_partially_covered_side_of_the_threshold(seed):
    """What the builder actually owes the planner. The realized ratio only approaches the
    target — prices are clamped into each item kind's own range so the receipt stays
    plausible — but the verdict it produces has to be the one that was asked for. Under
    the STRICT rule that is guaranteed by the existence of a non-covered line rather than
    by where the ratio lands, which is exactly why the clamp is tolerable."""
    receipt = mixed(seed)
    items = receipt.line_items
    covered = covered_total("vitamins_nutrition", items)
    every_line_covered = all(resolved_coverage("vitamins_nutrition", items))

    assert not every_line_covered
    assert verdict_for(covered, receipt.total, every_line_covered=every_line_covered) is (
        Verdict.PARTIALLY_COVERED
    )


@pytest.mark.parametrize("target", ["0.3", "0.5", "0.7", "0.9"])
def test_the_realized_coverage_tracks_the_requested_one(target):
    """Averaged over seeds, because a single basket is a draw. Within 0.15 of the target:
    the clamp on prices is what stops it being exact, and closer than that is not
    something a label depends on — the verdict turns on the non-covered line existing, not
    on the ratio."""
    wanted = Decimal(target)
    realized = [
        covered_total("vitamins_nutrition", r.line_items) / r.total
        for r in (mixed(seed, coverage_target=target) for seed in range(40))
    ]
    mean = sum(realized) / len(realized)
    assert abs(mean - wanted) < Decimal("0.15"), f"asked {wanted}, got {mean}"


@pytest.mark.parametrize("seed", range(20))
def test_a_mixed_basket_still_satisfies_every_document_invariant(seed):
    receipt = mixed(seed)
    assert validate_line_item_sum(receipt.line_items, receipt.total)
    assert validate_amount_in_words(receipt.amount_in_words, receipt.total)
    assert sum(line.gross for line in receipt.tax_lines) == receipt.total
    for item in receipt.line_items:
        assert validate_vat_letter(item.item_kind, item.vat_letter, "UA")
        assert item.price > 0


def test_the_non_covered_line_is_not_always_the_last_one():
    """Position is learnable. If every mixed receipt put its non-covered line at the
    bottom, a consumer could score well on this dataset without reading the line."""
    positions = set()
    for seed in range(40):
        items = mixed(seed).line_items
        positions |= {i for i, item in enumerate(items) if not item.covered}
    assert len(positions) > 1


def test_a_mixed_basket_is_deterministic_under_seed():
    assert mixed(7) == mixed(7)


def test_a_mixed_basket_refuses_to_guess_its_coverage():
    """The planner chose the verdict; a default here would let the builder choose one."""
    with pytest.raises(ValueError):
        build(1, covered_only=False)


@pytest.mark.parametrize("target", ["0", "1", "-0.5", "1.5"])
def test_a_coverage_target_outside_the_open_unit_interval_is_refused(target):
    """At 1 there is no non-covered line and at 0 there is no covered one; both are other
    verdicts, reached by other mechanisms."""
    with pytest.raises(ValueError):
        mixed(1, coverage_target=target)


def test_a_covered_only_basket_refuses_a_coverage_target():
    with pytest.raises(ValueError):
        build(1, covered_only=True, coverage_target=Decimal("0.7"))


def test_a_receipt_longer_than_the_cap_is_refused():
    with pytest.raises(ValueError):
        build(1, item_count=MAX_LINE_ITEMS + 1)


def test_a_basket_can_be_sized_up_to_the_cap():
    """The planner sizes a basket upward to overrun an annual limit. If the distinct-name
    draw ran out of attempts first, the overrun would silently not happen."""
    receipt = build(3, item_count=MAX_LINE_ITEMS)
    assert len(receipt.line_items) == MAX_LINE_ITEMS
    assert len({item.name for item in receipt.line_items}) == MAX_LINE_ITEMS


def test_only_printable_item_kinds_are_offered():
    """`medicine` and `cosmetics` templates all carry `{drug}` or `{brand}`, and neither
    has a vocabulary yet. They must be skipped, not raise: an item kind that cannot be
    printed today is a gap in `config/vendors.json`, not a bug in the draw."""
    excluded = category("vitamins_nutrition")["excluded_items"]
    printable = renderable_kinds(excluded)

    assert "medical_device" in printable
    assert "medicine" not in printable
    assert "cosmetics" not in printable
    assert set(printable) < set(excluded)


def test_estimated_line_value_is_inside_the_price_ranges_it_summarizes():
    """It sizes a basket before the basket is drawn, so it only has to be the right order
    of magnitude — but a value outside every range would mean it summarizes nothing."""
    value = estimated_line_value("vitamins_nutrition")
    assert Decimal("90.00") < value < Decimal("1500.00")


@pytest.mark.parametrize("seed", range(30))
def test_tax_lines_account_for_every_line_item(seed):
    """Each VAT group's gross is the sum of its lines, and the groups partition the
    receipt — so the tax block totals back to the receipt total."""
    receipt = build(seed)

    assert sum(line.gross for line in receipt.tax_lines) == receipt.total
    assert {line.letter for line in receipt.tax_lines} == {
        item.vat_letter for item in receipt.line_items
    }


@pytest.mark.parametrize("seed", range(30))
def test_vat_amount_is_the_tax_inside_the_gross(seed):
    """Ukrainian receipts print VAT-inclusive prices, so the tax is extracted from the
    gross rather than added to it: vat = gross − gross / (1 + rate/100)."""
    for line in build(seed).tax_lines:
        rate = Decimal(str(line.rate))
        expected = (line.gross - line.gross / (1 + rate / 100)).quantize(Decimal("0.01"))
        assert line.vat == expected
        assert line.vat < line.gross


def test_both_vat_groups_are_reachable():
    """A pharmacy basket mixes 7% and 20% lines. If every receipt came out with a single
    tax line, the tax block would never be exercised in the dataset."""
    group_counts = {len(build(seed).tax_lines) for seed in range(40)}
    assert max(group_counts) >= 2


# ------------------------------------------------------------------- fiscals --


def test_receipt_number_is_an_eleven_character_alphanumeric_id():
    """Modern ПРРО issue a short alphanumeric id, not a sequential number — a field a
    consumer has to extract, so the shape matters."""
    for seed in range(20):
        assert re.fullmatch(r"[A-Za-z0-9]{11}", build(seed).receipt_number)


def test_fiscal_device_number_is_ten_digits():
    value = build(1).fiscal_device_number
    assert len(value) == 10
    assert value.isdigit()


def test_title_may_carry_a_provider_suffix():
    titles = {build(seed).title for seed in range(40)}
    assert "ФІСКАЛЬНИЙ ЧЕК" in titles
    assert any(t.startswith("ФІСКАЛЬНИЙ ЧЕК ") for t in titles)


def test_acquiring_block_matches_the_configured_patterns():
    """Patterns are read from fiscal-rules.yaml rather than restated, so this test is
    what ties the generated values to the configuration. Restating them here would let
    the two drift apart while both looked correct."""
    patterns = {
        field["key"]: field["pattern"]
        for field in jurisdiction("UA")["acquiring_block"]["fields"]
        if "pattern" in field
    }
    assert patterns, "the UA acquiring block should declare patterns"

    for seed in range(10):
        acq = build(seed).acquiring
        for key, pattern in patterns.items():
            assert re.fullmatch(pattern, getattr(acq, key)), f"{key} does not match {pattern}"


def test_the_skeleton_always_pays_by_card():
    """A card payment is the more informative document: the acquiring block carries the
    RRN, which is the deduplication key. Cash receipts have no acquiring block at all
    and are a separate variation, not a coin flip inside this one."""
    receipt = build(3)
    assert receipt.acquiring is not None
    assert receipt.payment_method == "БЕЗГОТІВКОВА"


def test_qr_payload_is_the_tax_authority_verification_url():
    receipt = build(5)
    assert receipt.qr_payload.startswith("https://cabinet.tax.gov.ua/")
    assert receipt.receipt_number in receipt.qr_payload
    assert receipt.fiscal_device_number in receipt.qr_payload
    assert f"{receipt.total:.2f}" in receipt.qr_payload


def test_time_is_printed_with_dashes_not_colons():
    """ПРРО software prints 14-22-51, not 14:22:51."""
    assert build(1).render_context()["time"] == "14-22-51"


def test_decimal_separator_varies_between_receipts():
    """Vendors differ; both variants must appear or a consumer only ever sees one."""
    separators = {build(seed).decimal_separator for seed in range(40)}
    assert separators == {".", ","}


# -------------------------------------------------------------- ground truth --


def test_ground_truth_matches_the_receipt():
    receipt = build(9)
    truth = receipt.ground_truth(
        doc_id="p001_c1_d1",
        source_file="p001_c1_d1.png",
        capture=Capture.SCREENSHOT,
        field_bboxes={"total": (1.0, 2.0, 3.0, 4.0)},
    )

    assert truth.doc_type is DocType.FISCAL_RECEIPT
    assert truth.language == "uk"
    assert truth.currency == "UAH"
    assert truth.amount == receipt.total
    assert truth.date == date(2026, 8, 3)
    assert truth.counterparty == receipt.seller.name
    assert truth.line_items == receipt.line_items
    assert truth.field_bboxes == {"total": (1.0, 2.0, 3.0, 4.0)}


def test_ground_truth_carries_its_provenance():
    truth = build(9).ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert truth.synthetic is True
    assert truth.generator_version


def test_a_fiscal_receipt_declares_a_fiscal_qr_and_a_fiscal_number():
    """This archetype is the honest one. The traps that look like it but are not
    (templates 12 and 13) flip exactly these flags."""
    truth = build(9).ground_truth(
        doc_id="d1", source_file="d1.png", capture=Capture.SCREENSHOT, field_bboxes={}
    )
    assert truth.has_qr
    assert truth.qr_is_fiscal
    assert truth.has_fiscal_number


def test_line_item_names_carry_no_unresolved_placeholders():
    """Templates in policy.yaml contain {dose}, {n}, {brand}. A leftover brace would be
    printed literally onto the receipt image."""
    for seed in range(30):
        for item in build(seed).line_items:
            assert "{" not in item.name and "}" not in item.name
